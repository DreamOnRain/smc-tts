def deal_file():
    with open('filelists/aishell3_train.txt.cleaned', 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        line = line.strip()
        parts = line.split('|')
        parts[1] = str(int(parts[1]) + 1)
        processed_line = '|'.join(parts)
        processed_lines.append(processed_line)


    with open('aishell3_train.txt.cleaned', 'w') as file:
        for line in processed_lines:
            file.write(line + '\n')


import random

def shuffle_file_lines():
    input_file = 'aishell3_train.txt.cleaned'
    output_file = 'chinese_train.cleaned'
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def chec():
    with open('filelists/chinese_test.cleaned', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) != 3:
                print(parts)

if __name__ == "__main__":
    # shuffle_file_lines()
    # deal_file()
    chec()