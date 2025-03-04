import torch
import utils
from models import SynthesizerTrn
import soundfile as sf
from pypinyin import pinyin, Style
from chinese import get_pinyin_from_text, get_tones_from_text, init_chinese, get_stopwords
import time
from datetime import datetime
import jieba
import os
from generation import get_phonemes, get_text_by_cleaner, get_emotion
from cs_test import text_to_tensor, text_to_tensor2

PWD = os.path.dirname(os.path.realpath(__file__))

CONFIG = {
    'Chinese': {
        'config_file': PWD + "/configs/cscs.json",
        'checkpoint_file': PWD + "/checkpoints/cs3/G_720000.pth",
        'output_dir': PWD + "/outputs/cs/",
        'num_speakers': 220,
        'text_cleaners': "chinese_cleaners1",
        'sampling_rate': 16000,
    },
    'English': {
        'config_file': PWD + "/configs/cscs.json",
        'checkpoint_file': PWD + "/checkpoints/cs3/G_720000.pth",
        'output_dir': PWD + "/outputs/cs/",
        'num_speakers': 220,
        'text_cleaners': "chinese_cleaners1",
        'sampling_rate': 16000,
    },
}

def ocelot_init():
    jieba.initialize()
    _ = pinyin("初始化", style=Style.NORMAL)
    init_chinese()
    get_stopwords()

def ocelot_load_models():
    ocelot_init()

    # Load Chinese model
    hps = utils.get_hparams_from_file(CONFIG['Chinese']['config_file'])
    cs_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        n_pitches=hps.data.n_pitches,
        **hps.model).cuda()
    cs_model.eval()
    utils.load_checkpoint(CONFIG['Chinese']['checkpoint_file'], cs_model, None)

    return cs_model

def ocelot_infer(model, language, text, speaker_id, emotion):
    stn_tst, lang_ids = text_to_tensor2(text)
    stn_tst = stn_tst.cuda()
    lang_ids = lang_ids.cuda().unsqueeze(0).cuda()
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0).cuda()
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        model.to(x_tst.device)
        sid = torch.LongTensor([speaker_id]).cuda()
        pitch = torch.LongTensor([0]).cuda()
        inference_start_time = time.time()
        audio = model.infer(x_tst, x_tst_lengths, pitch=pitch, lang_ids=lang_ids, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        inference_time = time.time() - inference_start_time
        inference_time = f'{inference_time:.6f}'

    if not os.path.exists(CONFIG[language]['output_dir'] + f'{speaker_id}/'):
        os.makedirs(CONFIG[language]['output_dir'] + f'{speaker_id}/')

    filepath = CONFIG[language]['output_dir'] + f'{speaker_id}/{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
    sf.write(filepath, audio, CONFIG[language]['sampling_rate'])
    return filepath, 'xx', 'phonemes', 'tones', inference_time


def ocelot_generation(model, language, text, speaker_id, emotion, speed=1.0, resynthesis=True):
    if not resynthesis:
        return 'resynthesis is False', None, None, None, None, None


    filepath, sep_text, phonemes, tones, inference_time = ocelot_infer(model, language, text, speaker_id, emotion)
    message = 'success'

    return message, filepath, sep_text, phonemes, tones, inference_time

import sys

def main(language, text, speakerid, emotion='happy', speed=1.0):
    cs_model = ocelot_load_models()

    message, filepath, sep_text, phonemes, tones, inference_time = ocelot_generation(
        cs_model, language, text, int(speakerid), emotion, speed, True
    )
    
    print(f"Message: {message}\nFilepath: {filepath}\nSeparated Text: {sep_text}\nPhonemes: {phonemes}\nTones: {tones}\nInference Time: {inference_time}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <language> <text> [emotion] [speed]")
        sys.exit(1)

    language = sys.argv[1]
    text = sys.argv[2]
    speakerid = sys.argv[3]
    emotion = sys.argv[4] if len(sys.argv) > 3 else 'happy'
    speed = float(sys.argv[5]) if len(sys.argv) > 4 else 1.0
    main(language, text, speakerid, emotion, speed)
