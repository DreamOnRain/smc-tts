import random

input_file_path = 'filelists/csmsc_triple_phonemes.cleaned'
train_file_path = 'filelists/csmsc_train.cleaned'
valid_file_path = 'filelists/csmsc_valid.cleaned'

with open(input_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)

print(len(lines))
split_index = 9600
train_lines = lines[:split_index]
valid_lines = lines[split_index:]
print(len(train_lines), len(valid_lines))

with open(train_file_path, 'w', encoding='utf-8') as f:
    for line in train_lines:
        f.write(line)

with open(valid_file_path, 'w', encoding='utf-8') as f:
    for line in valid_lines:
        f.write(line)
