import os
import torch
from torch import nn
from torch.nn import functional as F
import pyworld as pw
import numpy as np
import torchaudio
import torchcrepe
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans


class PitchController(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim=7, embed_dim=10):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 512)

        self.time_series_path = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1 * 16, 512),
            nn.ReLU()
        )

        self.condition_embedding = nn.Embedding(condition_dim, embed_dim)
        self.condition_fc = nn.Linear(embed_dim, 512)
        self.fc_mean = nn.Linear(512 * 3, latent_dim)
        self.fc_logvar = nn.Linear(512 * 3, latent_dim)

    def forward(self, x, time_series_condition, class_condition):
        x_processed = torch.relu(self.input_fc(x))
        time_series_processed = self.time_series_path(time_series_condition)

        class_condition_embedded = self.condition_embedding(class_condition)
        class_condition_processed = torch.relu(self.condition_fc(class_condition_embedded))

        combined = torch.cat([x_processed, time_series_processed, class_condition_processed], dim=1)

        m = self.fc_mean(combined)
        logs = self.fc_logvar(combined)

        return m, logs


def calculate_pitch(file_dir):
    pitch_data = {}

    def process_file(filepath):
        pitch = calculate_pitch_average(filepath)
        return filepath, pitch

    filelist = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            filelist.append(os.path.join(root, file))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_file, filelist), total=len(filelist), desc="Calculating pitch"))

    for filepath, pitch in results:
        pitch_data[filepath] = pitch

    print(pitch_data)
    return pitch_data

def calculate_pitch_average(file_path):
    audio, sr = torchaudio.load(file_path)
    pitch, t = pw.dio(audio.squeeze(0).numpy().astype(np.float64), sr, frame_period=10)
    pitch = pitch[pitch > 0]
    if pitch.shape[0] == 0:
        return 0, 0, 0, 0, 0, 0
    mean = np.mean(pitch)
    variance = np.var(pitch)
    max_pitch = np.max(pitch)
    min_pitch = np.min(pitch)
    pitch_range = max_pitch - min_pitch
    trend = pitch[-1] - pitch[0]
    return mean, variance, max_pitch, min_pitch, pitch_range, trend

def calculate_durations(file_path, phonemes):
    audio, sr = torchaudio.load(file_path)
    audio_length = audio.shape[1] / sr
    phonemes_length = len(phonemes.split())
    duration_per_phoneme = audio_length / phonemes_length
    return duration_per_phoneme

def calculate_phonemes_per_duration(file_path, phonemes):
    audio, sr = torchaudio.load(file_path)
    audio_length = audio.shape[1] / sr
    phonemes_length = len(phonemes.split())
    phonemes_per_duration = phonemes_length / audio_length
    return phonemes_per_duration

def calculate_all(files):
    pitches = {}
    for file in files:
        pitch = calculate_pitch_average(file)
        key = file.split('/')[-1][:-4]
        pitches[key] = pitch
    return pitches

def wav_files(dir):
    wav_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                wav_files.append(full_path)
    return wav_files

def load_original_label():
    original_list = []
    with open('./chinese_train.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            # for i, part in enumerate(parts):
            #     parts[i] = part.replace('%', '').replace('$', '')
            original_list.append(parts)
    return original_list

def label_cleaned():
    AUDIO_DIR = '/data1/jiyuyu/csmsc/Wave/'
    # audio_files = wav_files(AUDIO_DIR)
    # pitch_data = calculate_all(audio_files)

    original_list = load_original_label()
    # original_list = original_list[]
    # for index in range(len(original_list)):
    #     original_list[index][0] = AUDIO_DIR + original_list[index][0][:7] + '/' + original_list[index][0] + '.wav'

    for item in original_list:
        filename = item[0]
        item.append(calculate_pitch_average(filename))

    df = pd.DataFrame(original_list, columns=['filename', 'phonemes', 'pitch'])
    discretizer = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform')
    df['pitch_class'] = discretizer.fit_transform(df[['pitch']]).astype(int)
    # df.drop(columns=['pitch'], inplace=True)
    df = df[['filename', 'pitch_class', 'phonemes']]
    df.to_csv('pitch_cleaned_all.txt', sep='|', index=False)

def label_cleaned_thread():
    original_list = load_original_label()

    def process_item(item):
        filename = item[0]
        pitch_avg = calculate_pitch_average(filename)
        return item + [pitch_avg]

    thread_num = 10
    results = []

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(process_item, item) for item in original_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    df = pd.DataFrame(results, columns=['filename', 'sid', 'phonemes', 'pitch'])
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    df['pitch_class'] = discretizer.fit_transform(df[['pitch']]).astype(int)

    df = df[['filename', 'sid', 'phonemes', 'pitch_class', 'pitch']]
    df.to_csv('chinese_pitch_cleaned.txt', sep='|', index=False)

def change_label():
    original_list = []
    with open('./chinese_pitch_test.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            original_list.append(parts)
    df = pd.DataFrame(original_list, columns=['filename', 'sid', 'phonemes', 'pitch_c', 'pitch', 'varance', 'max', 'min', 'range'])
    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    df['pitch_class'] = discretizer.fit_transform(df[['pitch']]).astype(int)
    df = df[['filename', 'sid', 'phonemes', 'pitch_class', 'pitch']]
    df.to_csv('chinese_pitch_cleaned.txt', sep='|', index=False)

def get_pitch_torch(audio, max_length):
    sr = 44100
    audio = audio.cpu()
    audio = audio.float()

    pitch = torchcrepe.predict(audio, sr)
    cur_length = pitch.shape[-1]

    padding_needed = (max_length - cur_length) // 2
    pitch = F.pad(pitch, (padding_needed, padding_needed), mode='reflect')

    return pitch

def process_files_with_thread_pool(thread_num=None):
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(label_cleaned)}

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

def deal_english_dataset():
    original_list = []
    with open('./ljs_audio_text_train_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[1].endswith('.') or parts[1].endswith('?') or parts[1].endswith('!'):
                parts[1] = parts[1][:-1]
            parts[1] = '< ' + parts[1] + ' >'
            new_parts = [parts[0], 0, parts[1]]
            original_list.append(new_parts)
        file.close()
    with open('./vctk_audio_sid_text_train_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[1] = int(parts[1]) + 1
            if parts[2].endswith('.') or parts[2].endswith('?') or parts[2].endswith('!'):
                parts[2] = parts[2][:-1]
            parts[2] = '< ' + parts[2] + ' >'
            original_list.append(parts)
        file.close()

    def process_item(item):
        filename = item[0]
        evg, var, max_pitch, min_pitch, range_pitch, trend = calculate_pitch_average(filename)
        return item + [evg, var, max_pitch, min_pitch, range_pitch, trend]

    thread_num = 26
    results = []

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(process_item, item) for item in original_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    df = pd.DataFrame(results, columns=['filename', 'sid', 'phonemes', 'pitch', 'varance', 'max', 'min', 'range', 'trend'])
    # discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='kmeans')
    # df['pitch_class'] = discretizer.fit_transform(df[['pitch', 'trend']]).astype(int)

    # df = df[['filename', 'sid', 'phonemes', 'pitch', 'varance', 'max', 'min', 'range', 'trend']]
    df.to_csv('english_pitch_train.cleaned', sep='|', index=False)

def deal_english():
    original_list = []
    with open('./ljs_audio_text_test_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[1] = '< ' + parts[1] + ' >'
            new_parts = [parts[0], 0, parts[1]]
            original_list.append(new_parts)
        file.close()
    with open('./vctk_audio_sid_text_test_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[1] = int(parts[1]) + 1
            parts[2] = '< ' + parts[2] + ' >'
            original_list.append(parts)
        file.close()

    def process_item(item):
        filename = item[0]
        phonemes = item[2]
        evg, var, max_pitch, min_pitch, range_pitch, trend = calculate_pitch_average(filename)
        phonemes_per_duration = calculate_durations(filename, phonemes)
        return item + [evg, var, max_pitch, min_pitch, range_pitch, trend, phonemes_per_duration]

    thread_num = 28
    results = []

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(process_item, item) for item in original_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    df = pd.DataFrame(results, columns=['filename', 'sid', 'phonemes', 'pitch', 'varance', 'max', 'min', 'range', 'trend', 'phonemes_per_duration'])
    df.to_csv('english_pitch_test.cleaned', sep='|', index=False)

def dee():
    original_list = []
    with open('./english_pitch_test.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[3] == '0.0':
                continue
            original_list.append(parts)
        file.close()

    def process_item(item):
        filename = item[0]
        phonemes = item[2]
        phonemes_per_duration = calculate_phonemes_per_duration(filename, phonemes)
        return item + [phonemes_per_duration]

    thread_num = 28
    results = []

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = [executor.submit(process_item, item) for item in original_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    df = pd.DataFrame(results, columns=['filename', 'sid', 'phonemes', 'pitch', 'varance', 'max', 'min', 'range', 'trend', 'duration_per_phonemes', 'phonemes_per_duration'])
    df.to_csv('english_pitch_test.cleaned', sep='|', index=False)

def de():
    original_list = []
    with open('./english_pitch_test.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[3] == '0.0':
                continue
            original_list.append(parts)
        file.close()
    df = pd.DataFrame(original_list, columns=['filename', 'sid', 'phonemes', 'pitch', 'varance', 'max', 'min', 'range', 'trend', 'duration_per_phonemes', 'phonemes_per_duration'])
    X = df[['range', 'trend', 'phonemes_per_duration']].values

    kmeans = KMeans(n_clusters=8, random_state=42)
    df['pitch_class'] = kmeans.fit_predict(X)
    print("聚类中心: \n", kmeans.cluster_centers_)
    # discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='kmeans')
    # df['pitch_class'] = discretizer.fit_transform(df[['pitch']]).astype(int)

    df = df[['filename', 'sid', 'phonemes', 'pitch_class', 'range', 'trend', 'phonemes_per_duration']]
    df.to_csv('kmeans_english_pitch_test.cleaned', sep='|', index=False)

def combine():
    original_list = []
    with open('./english_pitch_train.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[3] == '0.0':
                continue
            original_list.append(parts)
        file.close()
    with open('./english_pitch_test.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[3] == '0.0':
                continue
            original_list.append(parts)
        file.close()
    df = pd.DataFrame(original_list, columns=['filename', 'sid', 'phonemes', 'pitch', 'varance', 'max', 'min', 'range', 'trend', 'duration_per_phonemes', 'phonemes_per_duration'])
    X = df[['range', 'trend', 'phonemes_per_duration']].values

    kmeans = KMeans(n_clusters=8, random_state=42)
    df['pitch_class'] = kmeans.fit_predict(X)
    print("聚类中心: \n", kmeans.cluster_centers_)
    # discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='kmeans')
    # df['pitch_class'] = discretizer.fit_transform(df[['pitch']]).astype(int)

    df = df[['filename', 'sid', 'phonemes', 'pitch_class', 'range', 'trend', 'phonemes_per_duration']]
    df.to_csv('demo_kmeans_english_pitch.cleaned', sep='|', index=False)

def shuffle():
    original_list = []
    with open('./kmeans_english_pitch.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            if parts[3] == '0.0':
                continue
            original_list.append(parts)
        file.close()
    df = pd.DataFrame(original_list, columns=['filename', 'sid', 'phonemes', 'pitch_class', 'range', 'trend', 'phonemes_per_duration'])
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df_sampled = df_shuffled.iloc[:2500]
    df_remaining = df_shuffled.iloc[2500:]

    df_sampled.to_csv('kmeans_english_pitch_test.cleaned', sep='|', index=False, header=False)
    df_remaining.to_csv('kmeans_english_pitch_train.cleaned', sep='|', index=False, header=False)


if __name__ == '__main__':
    # label_cleaned_thread()
    # change_label()
    # calculate_pitch('./outputs/Chinese/')
    # print(calculate_pitch_average('/home/jiyuyu/workspace/vits_pitch/outputs/Chinese/0/请问多少钱.wav'))
    # deal_english()
    combine()
    # shuffle()