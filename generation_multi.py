import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols_zh
from text import text_to_sequence
import soundfile as sf
from pypinyin import lazy_pinyin, Style
from chinese import get_pinyin_from_text, get_tones_from_text, init_chinese
import time
import os
import re
from unidecode import unidecode
from phonemizer import phonemize
import argparse

PWD = os.path.dirname(os.path.realpath(__file__))

CONFIG = {
    'Chinese': {
        'config_file': PWD + "/configs/csmsc.json",
        'checkpoint_file': PWD + "/models/chinese_csmsc.pth",
        'output_dir': PWD + "/outputs/Chinese/",
        'num_speakers': 10
    },
    'Chinese_ms': {
        'config_file': PWD + "/configs/aishell3.json",
        'checkpoint_file': PWD + "/models/chinese_aishell3.pth",
        'output_dir': PWD + "/outputs/Chinese/",
        'num_speakers': 250
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

def init():
    init_chinese()

def check_speaker_id_valid(CONFIG_KEY, speaker_id):
    return speaker_id >= -1 and speaker_id <= CONFIG[CONFIG_KEY]['num_speakers']

def check_emotion_valid(emotion):
    #happy | angry | amazed | worried | smug | naughty
    return emotion in {'happy', 'angry', 'amazed', 'worried', 'smug', 'naughty'}

def check_filepath_exists(filepath):
    return os.path.exists(filepath)

def synthesize(language='Chinese', text='这是默认输入', speaker_id=-1, speed=1, emotion='happy', resynthesis=True):
    if speaker_id == -1:
        CONFIG_KEY = language
    else:
        CONFIG_KEY = language + '_ms'

    if not check_speaker_id_valid(CONFIG_KEY, speaker_id):
        return 'failed: invalid speaker id', None, None, None, None, None, None

    if not check_emotion_valid(emotion):
        return 'failed: invalid emotion', None, None, None, None, None, None

    dir_path = CONFIG[CONFIG_KEY]['output_dir'] + f'{speaker_id}/'
    filepath = dir_path + f'{text}.wav'
    if not resynthesis:
        if check_filepath_exists(filepath):
            return 'success', audio, dir_path + f'{text}.wav', sep_text, phonemes, tones, 0

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    audio, filepath, sep_text, phonemes, tones, inference_time = infer(CONFIG_KEY, language, text, speaker_id, emotion)
    message = 'success'

    return message, audio, filepath, sep_text, phonemes, tones, inference_time

def infer(CONFIG_KEY, language, text, speaker_id, emotion):
    inference_start_time = time.time()

    if language == 'Chinese':
        sep_text, phonemes = get_pinyin_from_text(text)
        tones = get_tones_from_text(text)
    else:
        sep_text = text
        phonemes = get_phonemes(language, text)
        tones = phonemes

    init()
    hps = utils.get_hparams_from_file(CONFIG[CONFIG_KEY]['config_file'])
    if speaker_id == -1:
        net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)#.cuda()
    else:
        net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).cuda()
    net_g.eval()
    # _ = utils.load_checkpoint(search_latest_checkpoint(), net_g, None)
    utils.load_checkpoint(CONFIG[CONFIG_KEY]['checkpoint_file'], net_g, None)

    stn_tst = get_text(phonemes, hps, language)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        net_g.to(x_tst.device)
        if speaker_id == -1:
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        else:
            sid = torch.LongTensor([speaker_id]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    filepath = CONFIG[CONFIG_KEY]['output_dir'] + f'{speaker_id}/{text}.wav'
    sf.write(filepath, audio, hps.data.sampling_rate)
    inference_time = time.time() - inference_start_time
    inference_time = f'{inference_time:.2f}'
    return audio, filepath, sep_text, phonemes, tones, inference_time


def str2bool(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input for the inference system.")

    parser.add_argument("language", type=str, nargs='?', default="Chinese", help="The language to be used. 'Chinese' | 'English'. Default is 'Chinese'")
    parser.add_argument("text", type=str, help="The input text.")
    parser.add_argument("speaker_id", type=int, nargs='?', default=-1, help="The ID of the speaker. Default is 0.")
    parser.add_argument("emotion", type=str, nargs='?', default="happy", help="The emotion to be used. Default is 'happy'.")
    parser.add_argument("resynthesis", type=str2bool, nargs='?', default=True, help="Whether to perform resynthesis. Default is True.")

    args = parser.parse_args()

    print(synthesize(args.language, args.text, args.speaker_id, args.emotion, args.resynthesis))