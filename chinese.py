import re
import jieba
import os
import pypinyin
from pypinyin import lazy_pinyin, Style, pinyin, load_phrases_dict
from pypinyin.contrib.tone_convert import to_normal, to_tone, to_initials, to_finals, to_finals_tone3
import json

PWD = os.path.dirname(os.path.realpath(__file__))
STOPWORDS_FILE = PWD + "/stopwords.txt"
USER_PINYIN_DICT_FILE = PWD + "/user_pinyin_dict.json"
USER_JIEBA_DICT_FILE = PWD + "/user_jieba_dict.json"

stopwords = set()

comma_map = {
    '（': '',
    '）': '',
    '“': '',
    '”': '',
    '，': '#3',
    '、': '#3',
    '：': '#3',
    '——': '#3',
    '—': '#3',
    '；': '#3',
    '…': '#3',
    '。': '#5',
    '？': '#5',
    '！': '#5',
}

def replace_comma(chinese_line):
    chinese_line = chinese_line.replace('（', ' ')
    chinese_line = chinese_line.replace('）', ' ')
    chinese_line = chinese_line.replace('，', '')
    chinese_line = chinese_line.replace('、', '')
    chinese_line = chinese_line.replace('：', '')
    chinese_line = chinese_line.replace('——', '')
    chinese_line = chinese_line.replace('—', '')
    chinese_line = chinese_line.replace('；', '')
    chinese_line = chinese_line.replace('。', '')
    chinese_line = chinese_line.replace('“', '')
    chinese_line = chinese_line.replace('”', '')
    chinese_line = chinese_line.replace('？', '')
    chinese_line = chinese_line.replace('！', '')
    chinese_line = chinese_line.replace('…', '')
    while '  ' in chinese_line:
        chinese_line = chinese_line.replace('  ', ' ')
    return chinese_line

def replace_chinese_comma(chinese_line):
    chinese_line = chinese_line.replace('（', '#2')
    chinese_line = chinese_line.replace('）', '#2')
    chinese_line = chinese_line.replace('，', '#3')
    chinese_line = chinese_line.replace('、', '#3')
    chinese_line = chinese_line.replace('：', '#3')
    chinese_line = chinese_line.replace('——', '#3')
    chinese_line = chinese_line.replace('—', '#3')
    chinese_line = chinese_line.replace('；', '#3')
    chinese_line = chinese_line.replace('。', '#4')
    chinese_line = chinese_line.replace('“', '#3')
    chinese_line = chinese_line.replace('”', '#3')
    chinese_line = chinese_line.replace('？', '#4')
    chinese_line = chinese_line.replace('！', '#4')
    chinese_line = chinese_line.replace('…', '#3')
    while '  ' in chinese_line:
        chinese_line = chinese_line.replace('  ', ' ')
    return chinese_line

def process_line(chinese_line, pinyin_line):
    chinese_line = chinese_line.replace('#1', '#')
    chinese_line = chinese_line.replace('（', '#2')
    chinese_line = chinese_line.replace('）', '#2')
    chinese_line = chinese_line.replace('#2', '$')
    chinese_line = chinese_line.replace('#3', '^')
    chinese_line = chinese_line.replace('，', '#3')
    chinese_line = chinese_line.replace('、', '#3')
    chinese_line = chinese_line.replace('：', '#3')
    chinese_line = chinese_line.replace('——', '#3')
    chinese_line = chinese_line.replace('—', '#3')
    chinese_line = chinese_line.replace('；', '#3')
    chinese_line = chinese_line.replace('#3', '%')
    chinese_line = chinese_line.replace('#4', '&')
    # chinese_line = chinese_line.replace('#5', '&')
    chinese_line = chinese_line.replace('。', '')
    chinese_line = chinese_line.replace('“', '#4')
    chinese_line = chinese_line.replace('”', '#4')
    chinese_line = chinese_line.replace('？', '?')
    chinese_line = chinese_line.replace('！', '!')
    chinese_line = chinese_line.replace('…', '')
    # chinese_line = ' '.join(chinese_line)

    pinyin_line = pinyin_line.split()
    pause_set = {'#', '$', '%', '^', '&', '?', '!'}
    pause_count = 0
    chinese_line = process_pause_set(chinese_line)
    for char in chinese_line:
        if char in pause_set:
            pause_count += 1
    pinyin_new = [''] * len(chinese_line)
    count = 0
    if chinese_line.count('儿') > 1:
        print(chinese_line)
    if len(pinyin_line) != len(chinese_line) - pause_count:
        chinese_line = chinese_line.replace('儿', '')
    for index, part in enumerate(chinese_line):
        if part in pause_set:
            pinyin_new[index] = part
            count += 1
        else:
            pinyin_new[index] = pinyin_line[index - count]

    return f"{chinese_line}|< {' '.join(pinyin_new)} >"


def process_continue_pause(text):
    # text = replace_comma(text)
    max_num = 0
    new_text = []
    flag = False
    for char in text:
        if char == '#':
            flag = True
        if flag:
            if char.isdigit():
                max_num = max(max_num, int(char))
                continue
            elif char == '#':
                pass
                continue
            else:
                flag = False
                new_text.append(f'#{max_num}')
                max_num = 0
        new_text.append(char)
    return ''.join(new_text)

def process_pause_set(lst):
    pause_set = {'#', '$', '%', '^', '&', '?', '!', '>'}
    result = []
    i = 0
    while i < len(lst):
        if lst[i] in pause_set:
            j = i + 1
            while j < len(lst) and lst[j] == ' ':
                j += 1
            if j < len(lst) and lst[j] in pause_set:
                result.append(lst[j])
                i = j + 1
            else:
                result.append(lst[i])
                i += 1
        else:
            result.append(lst[i])
            i += 1
    return ''.join(result)

def process_txt_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        while True:
            chinese_line = infile.readline().strip()
            if not chinese_line:
                break
            pinyin_line = infile.readline().strip()
            line_number = chinese_line.split()[0]
            chinese_line = chinese_line[len(line_number):].strip()
            processed_line = process_line(chinese_line, pinyin_line)
            outfile.write(f"{line_number}|{processed_line}\n")


def add_prefix_to_file(input_file, output_file, prefix='aaaa'):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(f"{prefix}{line}")

def remove_middle_part(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            parts = line.strip().split('|')
            if len(parts) == 3:
                new_line = f"{parts[0]}|{parts[2]}\n"
                outfile.write(new_line)

def get_stopwords():
    global stopwords
    if stopwords:
        return stopwords
    print('Loading stopwords...')
    with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def add_ending_to_pinyin(pinyin):
    ending = {'.', '?', '!'}
    if pinyin[-1] == '>' and pinyin[-2] not in ending:
        pinyin = pinyin[:-1] + ['.'] + ['>']
    return pinyin

def get_tones_from_text(text, sandhi=False):
    tones = []
    pinyins = lazy_pinyin(text, style=Style.TONE3, tone_sandhi=sandhi, neutral_tone_with_five=True)
    for pinyin in pinyins:
        tones.append(to_tone(pinyin))
    return ' '.join(tones)

def get_phoneme_from_text(text, sandhi=False):
    stopwords = get_stopwords()
    seg_list = fenci_cutall(text)
    chinese_line = ' '.join(seg_list)
    pinyin_line = ' '.join(lazy_pinyin(chinese_line, style=Style.TONE3, tone_sandhi=sandhi, neutral_tone_with_five=True))
    pinyin_line = replace_comma(pinyin_line)

    new_seg_list = []
    for seg in seg_list:
        new_seg_list.append(seg)
        if seg in stopwords:
            new_seg_list.append('#1')

    chinese_line = '#1'.join(new_seg_list)
    chinese_line = replace_chinese_comma(chinese_line)
    chinese_line = chinese_line.replace(' ', '')
    chinese_line = process_continue_pause(chinese_line)
    # print(chinese_line, pinyin_line)
    processed_line = process_line(chinese_line, pinyin_line)
    phoneme = processed_line.split('|')[1]
    return chinese_line, phoneme

def get_pinyin_from_text(text: str):
    pause_set = {'#', '$', '%', '^', '&', '?', '!', '>', '<', '.'}
    new_line = []
    chinese_line, pinyin_line = get_phoneme_from_text(text)
    pinyin_line = pinyin_line.split()
    for index, pinyin in enumerate(pinyin_line):
        if pinyin in pause_set:
            new_line.append(pinyin)
            continue
        initial = to_initials(pinyin)
        final = to_finals_tone3(pinyin, neutral_tone_with_five=True)
        if final == '':
            print('Warning:', pinyin)
            continue
        if initial == '':
            initial = '~'
        new_line.append(f"{initial} {final}")
    # new_line = add_ending_to_pinyin(new_line) // for csmsc only
    return chinese_line, ' '.join(new_line)

def pinyin_convert(input_file):
    pause_set = {'#', '$', '%', '^', '&', '?', '!', '>', '<', '.'}
    new_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            chinese_line = parts[0]
            pinyin_line = parts[1]
            pinyin_line = pinyin_line.split()
            new_line = []
            for index, pinyin in enumerate(pinyin_line):
                if pinyin in pause_set:
                    new_line.append(pinyin)
                    continue
                initial = to_initials(pinyin)
                final = to_finals_tone3(pinyin)
                if final == '':
                    # print(pinyin)
                    continue
                if initial == '':
                    initial = '~'
                new_line.append(f"{initial} {final}")
            new_lines.append(f"{chinese_line}|{' '.join(new_line)}\n")

    with open('./output.txt', 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)

def max_split(s, substrings):
    n = len(s)
    dp = [[] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for substring in substrings:
            len_sub = len(substring)
            if i >= len_sub and s[i - len_sub:i] == substring and dp[i - len_sub] is not None:
                candidate = dp[i - len_sub] + [substring]
                if len(candidate) > len(dp[i]):
                    dp[i] = candidate

    res = dp[-1] if dp[-1] else None

    if res and sum(len(sub) for sub in res) != len(s):
        with open('error.txt', 'a', encoding='utf-8') as f:
            f.write(f"{s} {res}\n")
        return [s]
    return res

def fenci_cutall(text):
    seg_list = list(jieba.cut(text, cut_all=True))
    return max_split(text, seg_list)

def fenci_cutall_odd(text):
    seg_list = list(jieba.cut(text, use_paddle=True))
    result = []
    for seg in seg_list:
        cutall = jieba.cut(seg, cut_all=True)
        if cutall == seg:
            result.append(seg)
            continue
        else:
            word_list = sorted(cutall, key=len)
            i = 0
            while i < len(seg):
                for word in word_list:
                    if seg.startswith(word, i):
                        result.append(word)
                        i += len(word)
                        break
                else:
                    result = [text]
                    break

    return result

def is_chinese(text):
    return all(
        ('\u4e00' <= char <= '\u9fff') or
        ('\u3000' <= char <= '\u303f') or
        char in {'（', '）', '“', '”', '，', '、', '：', '——', '—', '；', '…', '。', '？', '！'}
        for char in text
    )

def is_english(text):
    return all(
        'a' <= char <= 'z' or
        'A' <= char <= 'Z' or
        char in {',', '.', '!', '?', '&', ' ', '-', '\''}
        for char in text
    )

def get_pinyin_with_heteronym(text):
    result = []
    for p in pinyin(text, heteronym=True):
        result.append(p[0])
    return result

def init_user_pinyin_dict():
    with open(USER_PINYIN_DICT_FILE, 'r', encoding='utf-8') as f:
        phrases_dict = json.load(f)
    load_phrases_dict(phrases_dict)

def init_user_jieba_dict():
    jieba.load_userdict(USER_JIEBA_DICT_FILE)

def init_chinese():
    init_user_pinyin_dict()
    init_user_jieba_dict()

if __name__ == '__main__':
    # phoneme = get_phoneme_from_text(text)
    # pinyin_convert('./input.txt')
    # print(get_tones_from_text('你好'))
    # print(list(jieba.cut('早上好', use_paddle=True)))
    # print(fenci_cutall('北京大学'))
    seg_list = list(jieba.cut('我不明白你的意思', cut_all=True))
    print(seg_list)
    print(max_split('我不明白你的意思', seg_list))