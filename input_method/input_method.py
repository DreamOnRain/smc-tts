import re

pinyin_map = {}

with open('jda_mtsu_ccfl.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if not line or line == "" or line.startswith('#'):
            continue

        columns = line.strip().split('\t')
        if len(columns) < 5:
            print(f"Error: {line}")
        hanzi = columns[1]
        frequency = int(columns[2])
        pinyin_column = columns[4]
        pinyin_list = pinyin_column.split('/')

        for pinyin in pinyin_list:
            pinyin_no_tone = pinyin_base = re.sub(r'\d+', '', pinyin)
            hanzi_tuple = (hanzi, frequency)
            if pinyin_no_tone in pinyin_map:
                for item in pinyin_map[pinyin_no_tone]:
                    if item[0] == hanzi:
                        break
                else:
                    pinyin_map[pinyin_no_tone].append(hanzi_tuple)
            else:
                pinyin_map[pinyin_no_tone] = [hanzi_tuple]

with open('pinyin_map_with_frequency_one_line.json', 'w', encoding='utf-8') as json_file:
    json_file.write("{\n")
    for pinyin, hanzi_list in pinyin_map.items():
        json_file.write(f'  "{pinyin}": [')
        json_file.write(", ".join([f'["{hanzi}", {frequency}]' for hanzi, frequency in hanzi_list]))
        json_file.write(" ],\n")
    json_file.write("}\n")
