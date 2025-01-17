import IPython.display as ipd
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols_zh
from text import text_to_sequence
import soundfile as sf
from pypinyin import lazy_pinyin, Style
from chinese import get_pinyin_from_text, get_tones_from_text
import time
import os
import sys
import re
from unidecode import unidecode
from phonemizer import phonemize

PWD = os.path.dirname(os.path.realpath(__file__))

CONFIG = {
    'Chinese': {
        'config_file': PWD + "/configs/chinese.json",
        'checkpoint_file': PWD + "/models/chinese.pth",
        'output_dir': PWD + "/outputs/Chinese/",
        'num_speakers': 220
    },
    'English': {
        'config_file': PWD + "/configs/ljs_base.json",
        'checkpoint_file': PWD + "/models/english_ljs.pth",
        'output_dir': PWD + "/outputs/English/",
        'num_speakers': 10
    },
    'English_ms': {
        'config_file': PWD + "/configs/vctk_base.json",
        'checkpoint_file': PWD + "/models/english_vctk.pth",
        'output_dir': PWD + "/outputs/English/",
        'num_speakers': 109
    }
}

SAMPLE_TEXT = "你好，这里是新加坡国立大学"
SAMPLE_PHONEME = " ni2 hao3 zhe4 li3 shi4 xin1 jia1 po1 guo2 li4 da4 xue2 "

# def search_latest_checkpoint():
#     checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('G_') and f.endswith('.pth')]
#     checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
#     return os.path.join(CHECKPOINT_DIR, checkpoints[-1])
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def collapse_whitespace(text):
  return re.sub(re.compile(r'\s+'), ' ', text)

def get_text(text, hps, language):
    text_norm = text_to_sequence(text, hps.data.text_cleaners, language, phoneme=True)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_phonemes(language, text):
    if language == 'Chinese':
        phonemes = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        phonemes = phonemes.replace('5','')
        phonemes = ' ' + phonemes + ' '
    else:
        text = unidecode(text)
        text = text.lower()
        text = expand_abbreviations(text)
        phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
        phonemes = collapse_whitespace(phonemes)
    return phonemes

def infer(language, text, speaker_id):
    inference_start_time = time.time()
    hps = utils.get_hparams_from_file(CONFIG[language]['config_file'])
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    # _ = utils.load_checkpoint(search_latest_checkpoint(), net_g, None)
    utils.load_checkpoint(CONFIG[language]['checkpoint_file'], net_g, None)

    if language == 'Chinese':
        sep_text, phonemes = get_pinyin_from_text(text)
        tones = get_tones_from_text(text)
        stn_tst = get_text(phonemes, hps, language)
    else:
        sep_text = text
        phonemes = get_phonemes(language, text)
        phonemes = 'h eɪ'
        tones = phonemes
        stn_tst = get_text(phonemes, hps, language)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([speaker_id]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    filepath = CONFIG[language]['output_dir'] + f'{text}.wav'
    sf.write(filepath, audio, hps.data.sampling_rate)
    inference_time = time.time() - inference_start_time
    inference_time = f'{inference_time:.2f}'
    return audio, filepath, sep_text, phonemes, tones, inference_time

if __name__ == "__main__":
    language = sys.argv[1]
    text = sys.argv[2]
    speaker_id = int(sys.argv[3])
    print(infer(language, text, speaker_id))