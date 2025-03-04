import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import attentions
import commons

class SingleLanguageTextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    # ！词汇表转向量
    self.emb = nn.Embedding(n_vocab, hidden_channels)

    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    # Transformer
    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.cbhg = CBHG(hidden_channels, hidden_channels)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask

  @staticmethod
  def sequence_mask(lengths, max_len):
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]

class MultiLanguageTextEncoder(nn.Module):
  def __init__(self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout):
    super().__init__()
    self.zh_encoder = SingleLanguageTextEncoder(
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    )
    self.en_encoder = SingleLanguageTextEncoder(
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    )
    self.lang_embedding = LanguageEmbedding(num_languages=2, embedding_dim=hidden_channels)
    self.emb = nn.Embedding(n_vocab, hidden_channels)

  def forward(self, x, x_lengths, lang_ids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang_ids = lang_ids.to(device)

    x_zh, m_zh, logs_zh, mask_zh = self.zh_encoder(x, x_lengths)
    x_en, m_en, logs_en, mask_en = self.en_encoder(x, x_lengths)

    xx = torch.full_like(x_zh, 0)
    mm = torch.full_like(m_zh, 0)
    logs = torch.full_like(logs_zh, 0)
    x_mask = torch.full_like(mask_zh, 0)

    mask = (lang_ids.unsqueeze(1).expand(-1, x_zh.size(1), -1) == 1).float()

    xx = x_zh + (x_en - x_zh) * mask
    mm = m_zh + (m_en - m_zh) * mask
    logs = logs_zh + (logs_en - logs_zh) * mask
    x_mask = mask_zh + (mask_en - mask_zh) * (lang_ids.unsqueeze(1).expand(-1, mask_en.size(1), -1) == 1).float()

    return xx, mm, logs, x_mask


class CBHG(nn.Module):
  def __init__(self, idim, odim, conv_bank_layers=8, conv_bank_chans=128,
                conv_proj_filts=3, conv_proj_chans=256, highway_layers=4,
                highway_units=128, gru_units=256):
    super(CBHG, self).__init__()
    self.idim = idim
    self.odim = odim
    self.conv_bank_layers = conv_bank_layers

    self.conv_bank = nn.ModuleList()
    for k in range(1, conv_bank_layers + 1):
        if k % 2 != 0:
            padding = (k - 1) // 2
        else:
            padding = ((k - 1) // 2, (k - 1) // 2 + 1)
        self.conv_bank.append(
            nn.Sequential(
                nn.ConstantPad1d(padding, 0.0),
                nn.Conv1d(idim, conv_bank_chans, k, stride=1, padding=0, bias=True),
                nn.BatchNorm1d(conv_bank_chans),
                nn.ReLU(),
            )
        )

    self.max_pool = nn.Sequential(
        nn.ConstantPad1d((0, 1), 0.0), nn.MaxPool1d(2, stride=1)
    )

    self.projections = nn.Sequential(
        nn.Conv1d(conv_bank_chans * conv_bank_layers, conv_proj_chans, conv_proj_filts, stride=1,
                    padding=(conv_proj_filts - 1) // 2, bias=True),
        nn.BatchNorm1d(conv_proj_chans),
        nn.ReLU(),
        nn.Conv1d(conv_proj_chans, idim, conv_proj_filts, stride=1,
                    padding=(conv_proj_filts - 1) // 2, bias=True),
        nn.BatchNorm1d(idim),
    )

    self.highways = torch.nn.ModuleList()
    self.highways += [torch.nn.Linear(idim, highway_units)]
    for _ in range(highway_layers):
      self.highways += [HighwayNet(highway_units)]

    self.gru = nn.GRU(highway_units, gru_units // 2, num_layers=1, batch_first=True, bidirectional=True)

    self.output = nn.Linear(gru_units, odim, bias=True)

  def forward(self, xs, ilens):
    xs = xs.transpose(1, 2)
    # lang_emb_conv = lang_emb_conv.transpose(1, 2)
    # lang_emb_gate = lang_emb_gate.transpose(1, 2)
    # lang_emb_gru = lang_emb_gru.transpose(1, 2)

    convs = []
    for k in range(self.conv_bank_layers):
        convs += [self.conv_bank[k](xs)]
    convs = torch.cat(convs, dim=1)  # (B, #CH * #BANK, Tmax)
    convs = self.max_pool(convs)
    convs = self.projections(convs).transpose(1, 2)  # (B, Tmax, idim)
    xs = xs.transpose(1, 2) + convs

    for highway in self.highways:
        xs = highway(xs)

    xs = pack_padded_sequence(xs, ilens, batch_first=True, enforce_sorted=False)
    xs, _ = self.gru(xs)
    xs, _ = pad_packed_sequence(xs, batch_first=True)

    return self.output(xs)


class HighwayNet(nn.Module):
  def __init__(self, idim):
    super(HighwayNet, self).__init__()
    self.idim = idim
    self.projection = torch.nn.Sequential(
        torch.nn.Linear(idim, idim), torch.nn.ReLU()
    )
    self.gate = nn.Linear(idim, idim)
    self.lang_fc = nn.Linear(idim, idim)
    self.softsign = nn.Softsign()

  def forward(self, x, lang_emb_gate):
    lang_emb_transformed = self.softsign(self.lang_fc(lang_emb_gate))
    proj = self.projection(x)
    gate = self.gate(x + lang_emb_transformed)
    gate = torch.sigmoid(gate)
    return proj * gate + x * (1.0 - gate)


class LanguageEmbedding(nn.Module):
    def __init__(self, num_languages, embedding_dim):
        super(LanguageEmbedding, self).__init__()
        self.num_languages = num_languages
        self.embedding_dim = embedding_dim

        self.embedding_zh = nn.Embedding(num_embeddings=5000, embedding_dim=embedding_dim)
        self.embedding_en = nn.Embedding(num_embeddings=5000, embedding_dim=embedding_dim)

        self.fc_conv = nn.Linear(embedding_dim, embedding_dim)
        self.fc_gate = nn.Linear(embedding_dim, embedding_dim)
        self.fc_gru = nn.Linear(embedding_dim, embedding_dim)

        self.softsign = nn.Softsign()

    def forward(self, lang_ids, text_ids, seq_len):
        device = self.embedding_zh.weight.device
        lang_ids = lang_ids.to(device)
        text_ids = text_ids.to(device)

        lang_emb_zh = self.embedding_zh(text_ids)
        lang_emb_en = self.embedding_en(text_ids)

        lang_mask_zh = (lang_ids == 0).float().unsqueeze(-1)
        lang_mask_en = (lang_ids == 1).float().unsqueeze(-1)

        lang_emb = lang_emb_zh * lang_mask_zh + lang_emb_en * lang_mask_en

        lang_emb = lang_emb.squeeze(1).expand(-1, seq_len, -1)

        lang_emb_conv = self.softsign(self.fc_conv(lang_emb))
        lang_emb_gate = self.softsign(self.fc_gate(lang_emb))
        lang_emb_gru = self.softsign(self.fc_gru(lang_emb))
        print(f"xs.shape: {text_ids.shape}, lang_emb_conv.shape: {lang_emb_conv.shape}")

        return lang_emb_conv, lang_emb_gate, lang_emb_gru
