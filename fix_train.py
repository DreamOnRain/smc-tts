def fix():
    save_set = {0, 8, 26, 28, 52, 77, 79}
    retain_list = []
    with open('english_emotion_test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            filepath = line.split('|')[0]
            sid = int(line.split('|')[1])
            phoneme = line.split('|')[2]
            emotion = line.split('|')[3]
            if sid not in save_set:
                continue

            retain_list.append(line)

    with open('english_emotion_test_fix.txt', 'w') as f:
        for line in retain_list:
            f.write(line)


if __name__ == '__main__':
    fix()