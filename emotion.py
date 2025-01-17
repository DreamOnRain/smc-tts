import re
import random
from unidecode import unidecode
from phonemizer import phonemize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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

def get_phonemes(text):
    text = unidecode(text)
    text = text.lower()
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    phonemes = '< ' + phonemes + ' >'
    return phonemes

emotions_dict = {
    'Angry': 0,
    'Happy': 1,
    'Neutral': 2,
    'Sad': 3,
    'Surprise': 4,
}

#happy | angry | amazed | worried | smug | naughty
emotions_d = {
    'nautral': 0,
    'happy': 1,
    'angry': 2,
    'surprise': 3,
    'sad': 4,
}

def deal_esd(filepath, sid):
    lst = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.strip().split('\t')
            part1 = 'DUMMY7/' + parts[0] + '.wav'
            if index % 100 == 0:
                print(f'Processing {index} lines')
                print(line)
            part2 = get_phonemes(parts[1])
            part3 = emotions_dict[parts[2]]
            new_parts = [part1, sid, part2, str(part3)]
            lst.append('|'.join(new_parts))
        file.close()
    return lst


def deal_esd_thread():
    original_list = []
    filepathlist= [
        '/data1/jiyuyu/esd-16000hz/0020.txt',
    ]

    for filepath in filepathlist:
        original_list.extend(deal_esd(filepath, '120'))

    with open('./processed.txt', 'a', encoding='utf-8') as file:
        for item in original_list:
            file.write(item + '\n')
        file.close()

def combine_all():
    original_list = []
    with open('./ljs_audio_text_train_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[0] = 'DUMMY8/' + parts[0].split('/')[-1]
            parts[1] = '< ' + parts[1] + ' >'
            new_parts = [parts[0], '0', parts[1], str(emotions_dict['Neutral'])]
            original_list.append('|'.join(new_parts))
        file.close()
    with open('./vctk_audio_sid_text_train_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[0] = 'DUMMY9/' + parts[0].split('/')[-1]
            parts[1] = str(int(parts[1]) + 1)
            parts[2] = '< ' + parts[2] + ' >'
            new_parts = [parts[0], parts[1], parts[2], str(emotions_dict['Neutral'])]
            original_list.append('|'.join(new_parts))
        file.close()
    with open('./ljs_audio_text_test_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[0] = 'DUMMY8/' + parts[0].split('/')[-1]
            parts[1] = '< ' + parts[1] + ' >'
            new_parts = [parts[0], '0', parts[1], str(emotions_dict['Neutral'])]
            original_list.append('|'.join(new_parts))
        file.close()
    with open('./vctk_audio_sid_text_test_filelist.txt.cleaned', 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|')
            parts[0] = 'DUMMY9/' + parts[0].split('/')[-1]
            parts[1] = str(int(parts[1]) + 1)
            parts[2] = '< ' + parts[2] + ' >'
            new_parts = [parts[0], parts[1], parts[2], str(emotions_dict['Neutral'])]
            original_list.append('|'.join(new_parts))
        file.close()
    with open('./processed.txt', 'a', encoding='utf-8') as file:
        for item in original_list:
            file.write(item + '\n')
        file.close()

def shuffle_txt():
    with open('./processed.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    part1 = lines[:3000]
    part2 = lines[3000:]

    with open('./english_emotions_test.cleaned', 'w', encoding='utf-8') as file:
        for line in part1:
            file.write(line)

    with open('./english_emotions_train.cleaned', 'w', encoding='utf-8') as file:
        for line in part2:
            file.write(line)

def check(split="|"):
    with open('./filelists/english_emotion_test.cleaned', encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    for line in filepaths_and_text:
        if len(line) != 4:
            print(line)
    return

def change_label():
    with open('./filelists/english_emotion_train.cleaned', encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split('|') for line in f]
    for line in filepaths_and_text:
        if line[3] == '0':
            line[3] = 'angry'
        elif line[3] == '1':
            line[3] = 'happy'
        elif line[3] == '2':
            line[3] = 'neutral'
        elif line[3] == '3':
            line[3] = 'sad'
        elif line[3] == '4':
            line[3] = 'surprise'
    with open('./english_emotion_train.txt', 'w', encoding='utf-8') as file:
        for item in filepaths_and_text:
            file.write('|'.join(item) + '\n')
        file.close()

    return

if __name__ == '__main__':
    change_label()