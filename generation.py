import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols_zh
from text import text_to_sequence
import soundfile as sf
from pypinyin import pinyin, lazy_pinyin, Style
from chinese import get_pinyin_from_text, get_tones_from_text, init_chinese
import time
import os
import re
from unidecode import unidecode
from phonemizer import phonemize
import argparse
from datetime import datetime
import jieba

PWD = os.path.dirname(os.path.realpath(__file__))

CONFIG = {
    'Chinese': {
        'config_file': PWD + "/configs/chinese.json",
        'checkpoint_file': PWD + "/models/chinese.pth",
        'output_dir': PWD + "/outputs/Chinese/",
        'num_speakers': 220,
        'text_cleaners': "chinese_cleaners1",
        'sampling_rate': 44100,
    },
    'English': {
        'config_file': PWD + "/configs/english_emotion.json",
        'checkpoint_file': PWD + "/models/G_620000.pth", # PWD + "/models/english_emotion.pth",
        'output_dir': PWD + "/outputs/English/",
        'num_speakers': 120,
        'text_cleaners': "english_cleaners2",
        'sampling_rate': 16000,
    },
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

def get_text_by_cleaner(text, cleaner, language):
    text_norm = text_to_sequence(text, cleaner, language, phoneme=True)
    text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text(text, hps, language):
    text_norm = text_to_sequence(text, 'english_cleaner2', language, phoneme=True)
    if True:
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
        phonemes = '< ' + phonemes + ' >'
    return phonemes

def init():
    init_chinese()

def check_speaker_id_valid(CONFIG_KEY, speaker_id):
    return speaker_id >= -1 and speaker_id <= CONFIG[CONFIG_KEY]['num_speakers']

def check_emotion_valid(emotion):
    # happy | angry | amazed | worried | smug | naughty
    return emotion in {'happy', 'angry', 'amazed', 'worried', 'smug', 'naughty', 'neutral'}

def check_filepath_exists(filepath):
    return os.path.exists(filepath)

def get_emotion(emotion):
    # angry | happy | neutral | sad | surprise
    emotion_u2s_dict = {
        'happy': 'happy',
        'angry': 'angry',
        'amazed': 'surprise',
        'worried': 'sad',
        'smug': 'neutral',
        'naughty': 'neutral',
        'neutral': 'neutral',
    }
    emotion_e2n_dict = {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3,
        'surprise': 4,
    }
    return emotion_e2n_dict[emotion_u2s_dict[emotion]]

def synthesize(language='Chinese', text='这是默认输入', speaker_id=0, emotion='happy', speed=1.0, resynthesis=True):
    CONFIG_KEY = language

    if not check_speaker_id_valid(CONFIG_KEY, speaker_id):
        return 'failed: invalid speaker id' + str(speaker_id), None, None, None, None, None, None

    if not check_emotion_valid(emotion):
        return 'failed: invalid emotion' + emotion, None, None, None, None, None, None
    pitch = get_emotion(emotion)

    dir_path = CONFIG[CONFIG_KEY]['output_dir'] + f'{speaker_id}/'
    filepath = dir_path + f'{text}.wav'
    if not resynthesis:
        return 'resynthesis is False', None, None, None, None, None, None

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    audio, filepath, sep_text, phonemes, tones, inference_time = load_infer(CONFIG_KEY, language, text, speaker_id, pitch, emotion)
    message = 'success'

    return message, audio, filepath, sep_text, phonemes, tones, inference_time

def load_infer(CONFIG_KEY, language, text, speaker_id, pitch, emotion):
    if language == 'Chinese':
        sep_text, phonemes = get_pinyin_from_text(text)
        tones = get_tones_from_text(text)
    else:
        sep_text = text
        phonemes = get_phonemes(language, text)
        tones = phonemes

    init()
    hps = utils.get_hparams_from_file(CONFIG[CONFIG_KEY]['config_file'])
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_pitches=hps.data.n_pitches,
        **hps.model).cuda()
    net_g.eval()
    # _ = utils.load_checkpoint(search_latest_checkpoint(), net_g, None)
    utils.load_checkpoint(CONFIG[CONFIG_KEY]['checkpoint_file'], net_g, None)

    stn_tst = get_text(phonemes, hps, language)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        net_g.to(x_tst.device)
        sid = torch.LongTensor([speaker_id]).cuda()
        pitch = torch.LongTensor([pitch]).cuda()
        if language == 'Chinese':
            pitch = None

        inference_start_time = time.time()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, pitch=pitch, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        inference_time = time.time() - inference_start_time
        inference_time = f'{inference_time:.2f}'

    # filepath = CONFIG[CONFIG_KEY]['output_dir'] + f'{speaker_id}/{text}.wav'
    filepath = CONFIG[CONFIG_KEY]['output_dir'] + f'{speaker_id}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
    sf.write(filepath, audio, hps.data.sampling_rate)
    return audio, filepath, sep_text, phonemes, tones, inference_time


def ocelot_init():
    jieba.initialize()
    _ = pinyin("初始化", style=Style.NORMAL)
    init()

def ocelot_load_model():
    ocelot_init()

    # Load Chinese model
    hps = utils.get_hparams_from_file(CONFIG['Chinese']['config_file'])
    chinese_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_pitches=hps.data.n_pitches,
        **hps.model).cuda()
    chinese_model.eval()
    utils.load_checkpoint(CONFIG['Chinese']['checkpoint_file'], chinese_model, None)

    # Load English model
    hps = utils.get_hparams_from_file(CONFIG['English']['config_file'])
    english_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_pitches=hps.data.n_pitches,
        **hps.model).cuda()
    english_model.eval()
    utils.load_checkpoint(CONFIG['English']['checkpoint_file'], english_model, None)

    return chinese_model, english_model

def ocelot_infer_chinese(model, text, speaker_id):
    sep_text, phonemes = get_pinyin_from_text(text)
    tones = get_tones_from_text(text)
    stn_tst = get_text(phonemes, CONFIG['Chinese']['text_cleaners'], 'Chinese')
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        model.to(x_tst.device)
        sid = torch.LongTensor([speaker_id]).cuda()
        pitch = None
        inference_start_time = time.time()
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, pitch=pitch, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        inference_time = time.time() - inference_start_time
        inference_time = f'{inference_time:.6f}'

    filepath = CONFIG['Chinese']['output_dir'] + f'{speaker_id}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
    sf.write(filepath, audio, 44100)
    return audio, filepath, sep_text, phonemes, tones, inference_time

def ocelot_infer_english(model, text, speaker_id, emotion):
    sep_text = text
    phonemes = get_phonemes('English', text)
    phonemes = "'h'he'eɪˌ"
    tones = phonemes
    stn_tst = get_text(phonemes, CONFIG['English']['text_cleaners'], 'English')
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        model.to(x_tst.device)
        sid = torch.LongTensor([speaker_id]).cuda()
        pitch = torch.LongTensor([0]).cuda()
        inference_start_time = time.time()
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, pitch=pitch, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        inference_time = time.time() - inference_start_time
        inference_time = f'{inference_time:.6f}'

    filepath = 'output.wav'#CONFIG['English']['output_dir'] + f'{speaker_id}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
    sf.write(filepath, audio, 16000)
    return audio, filepath, sep_text, phonemes, tones, inference_time

def ocelot_infer(model, language, text, speaker_id, emotion):
    if language == 'Chinese':
        sep_text, phonemes = get_pinyin_from_text(text)
        tones = get_tones_from_text(text)
        stn_tst = get_text(phonemes, CONFIG['Chinese']['text_cleaners'], 'Chinese')
    else:
        sep_text = text
        phonemes = get_phonemes('English', text)
        phonemes = "'h'e'eɪˌ"
        tones = phonemes
        stn_tst = get_text(phonemes, CONFIG['English']['text_cleaners'], 'English')
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        model.to(x_tst.device)
        sid = torch.LongTensor([speaker_id]).cuda()
        if language == 'Chinese':
            pitch = None
        else:
            pitch = torch.LongTensor([pitch]).cuda()
        inference_start_time = time.time()
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, pitch=pitch, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        inference_time = time.time() - inference_start_time
        inference_time = f'{inference_time:.6f}'

    filepath = 'output.wav'#CONFIG[language]['output_dir'] + f'{speaker_id}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
    sf.write(filepath, audio, CONFIG[language]['sampling_rate'])
    return audio, filepath, sep_text, phonemes, tones, inference_time


def ocelot_generation(model, language, text, speaker_id, emotion, speed, resynthesis):
    if language == 'Chinese':
        audio, filepath, sep_text, phonemes, tones, inference_time = ocelot_infer_chinese(model, text, speaker_id)
    else:
        audio, filepath, sep_text, phonemes, tones, inference_time = ocelot_infer_english(model, text, speaker_id, emotion)
    message = 'success'
    return message, filepath, sep_text, phonemes, tones, inference_time


def str2bool(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        return True

if __name__ == "__main__":
    chinese_model, english_model = ocelot_load_model()
    # ocelot_generation(english_model, 'Chinese', '你好', 0, 'happy', 1.0, True)
    ocelot_generation(english_model, 'English', 'hey', 0, 'worried', 1.0, True)