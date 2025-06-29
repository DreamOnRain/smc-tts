from cs import ocelot_generation, ocelot_load_models
import soundfile as sf
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.inference.interfaces import foreign_class
from speechbrain.utils.metric_stats import ErrorRateStats
from chinese import fenci_cutall
import os
from tqdm import tqdm
import random
from pypinyin import lazy_pinyin, Style

text_inputs = []
audio_inputs = []
with open('aishell1_sentences.txt', 'r') as f:
    for line in f.readlines():
        audio_inputs.append(line.split(' ')[0])
        text_inputs.append(line.split(' ')[1])

all_texts = random.sample(text_inputs, 1000)
all_texts = [text for text in all_texts]

with open('test_sentences.txt', 'w') as f:
    f.write('\n'.join(all_texts))

sampling_rate = 16000

chinese_model = ocelot_load_models()
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-ctc-aishell", savedir="/data1/jiyuyu/speechbrain/pretrained_models/asr-wav2vec2-ctc-aishell")
asr_model = foreign_class(source="speechbrain/asr-wav2vec2-ctc-aishell",  pymodule_file="custom_interface.py", classname="CustomEncoderDecoderASR")

def find_wav_file(directory, target_file):
    for root, dirs, files in os.walk(directory):
        if target_file in files:
            return os.path.join(root, target_file)
    return None

def calculate_rtf(text_list):
    total_inference_time = 0
    total_audio_duration = 0
    wer_stats = ErrorRateStats()
    wer_stats_gt = ErrorRateStats()
    predicted_list = []
    gt_list = []

    for index, text_input in enumerate(tqdm(text_list, desc="Processing")):
        wav_input = find_wav_file('/data1/jiyuyu/AISHELL-1/data_aishell/wav/', audio_inputs[index] + '.wav')
        text_input = lazy_pinyin(text_input[0:-1], style=Style.TONE3, neutral_tone_with_five=True)
        text = '[' + '] ['.join(text_input) + ']'
        _, audio_output, *rest, inference_time = ocelot_generation(chinese_model, 'Chinese', text, 0, 0)
        
        with sf.SoundFile(audio_output) as audio_file:
            frames = audio_file.frames
            sampling_rate = audio_file.samplerate
            audio_duration = frames / sampling_rate

        try:
            predicted_text = asr_model.transcribe_file(audio_output)
            predicted_list.append(predicted_text)

            gt_text = asr_model.transcribe_file(wav_input)
            gt_list.append(gt_text)

        except Exception as e:
            print(f"Error processing audio for WER calculation: {e}")

        total_inference_time += float(inference_time)
        total_audio_duration += audio_duration

    average_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
    wer_stats.append(list(range(len(text_list))), predicted_list, text_list)
    wer_stats_gt.append(list(range(len(text_list))), gt_list, text_list)
    with open('./cer.txt', "w", encoding="utf-8") as w:
        wer_stats.write_stats(w)

    with open('./cer_gt.txt', "w", encoding="utf-8") as w:
        wer_stats_gt.write_stats(w)

    return average_rtf

def load_aishell1_sentences():
    with open('/data1/jiyuyu/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt', 'r') as f, open('aishell1_sentences.txt', 'w') as f2:
        lst = f.readlines()
        for i in range(len(lst)):
            wav = lst[i].split(' ')[0]
            lst[i] = lst[i].split(' ')[1:]
            lst[i] = wav + ' ' + ''.join(lst[i])
            f2.write(lst[i])
    return lst

def find_file_path(filename, search_path="."):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def ai():
    audio_list = []
    text_list = []
    with open('save_path', 'r') as f:
        lll = f.readlines()
        for lst in lll:
            audio_list.append(lst.split(' ')[0])
            text_list.append(lst.split(' ')[1])
    f.close()
    assert len(audio_list) == len(text_list)
    wer_stats = ErrorRateStats()
    predicted_list = []

    for audio_input in tqdm(audio_list, desc="Processing"):
        try:
            predicted_text = asr_model.transcribe_file(audio_input)
            predicted_list.append(predicted_text)
        except Exception as e:
            print(f"Error processing audio for WER calculation: {e}")
    wer_stats.append(list(range(len(text_list))), predicted_list, text_list)
    with open('./cer_aishell1.txt', "w", encoding="utf-8") as w:
        wer_stats.write_stats(w)

average_rtf = calculate_rtf(all_texts)

with open("RTF.txt", "w") as file:
    file.write(f"Overall Average RTF: {average_rtf}")
# load_aishell1_sentences()