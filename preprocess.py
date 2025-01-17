import argparse
import text
from utils import load_filepaths_and_text
from pypinyin.contrib.tone_convert import to_normal, to_tone, to_initials, to_finals, to_finals_tone3
from sklearn.model_selection import train_test_split
from chinese import fenci_cutall
import json

TRAIN_PATH = '/data1/jiyuyu/AISHELL-3/data_aishell3/train/wav/'
TEST_PATH = '/data1/jiyuyu/AISHELL-3/data_aishell3/test/wav/'

def process_chinese_label_foraishell3(filename, split="|", train=True):
  processed_lines = []
  with open(filename, encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split(split)
      if len(parts) > 2:
        part1 = 'DUMMY3/' + f'{parts[0][:7]}/{parts[0]}' + '.wav'
        part2 = spk_id_list.index(parts[0][:7])
        chinese_list = list(parts[2])
        pinyin_list = parts[1].split()
        if pinyin_list[-1] != '$' and pinyin_list[-1] != '%' and pinyin_list[-1] != '#':
          pinyin_list.append('$')
        if chinese_list[-1] != '$' and chinese_list[-1] != '%' and chinese_list[-1] != '#':
          chinese_list.append('$')
        count = 0
        # while len(chinese_list) != len(pinyin_list):
        #   try:
        #     idx = chinese_list.index('儿')
        #     if chinese_list[idx - 1] == '%' or chinese_list[idx - 1] == '#' or chinese_list[idx - 1] == '$':
        #       chinese_list.pop(idx)
        #       idx -= (2 + count)
        #     else:
        #       chinese_list.pop(idx)
        #       idx -= (1 + count)
        #     count += 1
        #     if pinyin_list[idx][-2] == 'r':
        #       pinyin_list[idx] = pinyin_list[idx][:-2] + pinyin_list[idx][-1]
        #   except:
        #     print(len(chinese_list), len(pinyin_list))
        #     print(chinese_list, pinyin_list)
        #     exit(0)
        for idx, char in enumerate(pinyin_list):
          if len(char) >= 2 and char[0] != 'e' and char[1] != 2:
            if char[-2] == 'r':
              pinyin_list[idx] = char[:-2] + char[-1]

        new_line = []
        for index, p in enumerate(pinyin_list):
          if p == '%':
            new_line.append('#')
            continue
          if p == '$':
            if index != len(pinyin_list) - 1:
              new_line.append('^')
            else:
              pass
            continue
          initial = to_initials(p)
          if initial == '':
            initial = '~'
          final = to_finals_tone3(p, neutral_tone_with_five=True)
          new_line.append(initial)
          new_line.append(final)

        part3 = '< ' + ' '.join(new_line) + ' >'
        nnn = f"{part1}|{part2}|{part3}"
        processed_lines.append(nnn)

  print(processed_lines[0])
  return processed_lines

def read_spk_info():
  id_list = []
  with open('spk-info.txt', 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('\t')
      id_list.append(parts[0])
  id_list.sort()
  return id_list

def spk_dict_to_json():
  id_list = read_spk_info()
  spk_dict = {}
  for idx, spk_id in enumerate(id_list):
    spk_dict[spk_id] = idx
  with open('spk_dict.json', 'w', encoding='utf-8') as f:
    json.dump(spk_dict, f, indent=4)

def load_spk_dict():
  data = {}
  lines = []
  with open('spk-info.txt', 'r', encoding='utf-8') as f:
    for line in f:
      lines.append(line)

    for idx, line in enumerate(sorted(lines)):
      parts = line.strip().split('\t')
      spk_id, letter, gender, accent = parts

      if gender not in data:
          data[gender] = {}
      if letter not in data[gender]:
          data[gender][letter] = {}
      if accent not in data[gender][letter]:
          data[gender][letter][accent] = []

      data[gender][letter][accent].append(idx + 1)

  json_data = json.dumps(data, indent=4, ensure_ascii=False)
  with open('output.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

def load_vctk_spk_dict():
  data = {}
  lines = []
  with open('speaker-info.txt', 'r', encoding='utf-8') as f:
    for line in f:
      lines.append(line)

    for idx, line in enumerate(sorted(lines)):
      parts = line.strip().split('  ')
      spk_id, ages, gender, accent, *rest = parts
      if gender not in data:
          data[gender] = {}
      if ages not in data[gender]:
          data[gender][ages] = {}
      if accent not in data[gender][ages]:
          data[gender][ages][accent] = []

      data[gender][ages][accent].append(idx + 1)
  
  json_data = json.dumps(data, indent=4, ensure_ascii=False)
  with open('output_vctk.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

def separate_chinese_pinyin(text):
    import re
    chinese_part = re.findall(r'[\u4e00-\u9fff]+', text)
    pinyin_part = re.findall(r'[a-z]+\d', text)
    
    chinese_str = ' '.join(chinese_part)
    pinyin_str = ' '.join(pinyin_part)
    
    return chinese_str, pinyin_str

def process_test():
  processed_lines = []
  with open('content.txt', 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('\t')
      # print(parts[1])
      chinese_str, pinyin_str = separate_chinese_pinyin(parts[1])

      chinese_list = chinese_str.split()
      pinyin_list = pinyin_str.split()

      for idx, char in enumerate(chinese_list):
        if '儿' in char:
          if pinyin_list[idx][-2] == 'r':
            pinyin_list[idx] = pinyin_list[idx][:-2] + pinyin_list[idx][-1]

      fenci = fenci_cutall(''.join(chinese_str.split()))
      chinese_line = '#'.join(fenci)
      # print(chinese_line)

      part1 = 'DUMMY4/' + f'{parts[0][:7]}/{parts[0]}'
      part2 = spk_id_list.index(parts[0][:7])
      count = 0
      new_line = []
      for index, p in enumerate(pinyin_list):
        if chinese_line[index + count] == '#':
          new_line.append('#')
          count += 1
        initial = to_initials(p)
        if initial == '':
          initial = '~'
        final = to_finals_tone3(p, neutral_tone_with_five=True)
        new_line.append(initial + ' ' + final)

      part3 = '< ' + ' '.join(new_line) + ' >'
      # print(part3)
      processed_lines.append(f"{part1}|{part2}|{part3}")
  
  print(processed_lines[0])
  return processed_lines

if __name__ == '__main__':
  # load_spk_dict()
  load_vctk_spk_dict()
  exit(0)
  spk_id_list = read_spk_info()
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/label_train-set.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

  args = parser.parse_args()

  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = process_chinese_label_foraishell3(filelist)
    # filepaths_and_text = []
    processed_lines = process_test()

    all_lines = filepaths_and_text + processed_lines

    def split_list(all_lines, train_ratio=25):
        test_ratio = 1 / (train_ratio + 1)
        train_list, test_list = train_test_split(all_lines, test_size=test_ratio)
        return train_list, test_list

    train_list, test_list = split_list(all_lines)

    with open("aishell3_train.txt.cleaned", "w", encoding="utf-8") as f:
      # f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
      f.writelines([x + '\n' for x in train_list])

    with open("aishell3_test.txt.cleaned", "w", encoding="utf-8") as f:
      f.writelines([x + '\n' for x in test_list])