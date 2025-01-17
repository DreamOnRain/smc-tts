import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import wavfile

def resample_wave(wav_in, wav_out, sample_rate):
    wav, _ = librosa.load(wav_in, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
    wavfile.write(wav_out, sample_rate, wav.astype(np.int16))


def process_file(file, wavPath, outPath, sr):
    # oriPath = file
    # file = file.split('/')[-1]
    if file.endswith(".wav"):
        file = file[:-4]
        resample_wave(f"{wavPath}/{file}.wav", f"{outPath}/{file}.wav", sr)


def process_files_with_thread_pool(wavPath, outPath, sr, thread_num=None):
    # files = []
    # for root, dirs, filenames in os.walk(wavPath):
    #     for f in filenames:
    #         if f.endswith(".wav"):
    #             files.append(root + '/' + f)
    files = [f for f in os.listdir(f"{wavPath}") if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(process_file, file, wavPath, outPath, sr): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {sr}'):
            future.result()

if __name__ == "__main__":
    process_files_with_thread_pool('/data1/jiyuyu/LJ-Speech/LJSpeech-1.1/wavs', '/data1/jiyuyu/ljspeech-16000hz/', 16000, 28)