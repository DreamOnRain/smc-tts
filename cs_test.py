import re
from cs_chinese import get_pinyin_from_text
from generation import get_tones_from_text, get_text_by_cleaner, get_phonemes
import torch
from pypinyin import lazy_pinyin, Style, pinyin, load_phrases_dict
from pypinyin.contrib.tone_convert import to_normal, to_tone, to_initials, to_finals, to_finals_tone3

def split_and_mark_language(text):
    result = []
    parts = re.split(r"(\[[^\]]*\])|([.,\s])", text)

    temp = []
    for part in parts:
        if part is None or part == "":
            continue
        elif part.startswith("[") and part.endswith("]"):
            result.append(["zh", part[1:-1]])
        else:
            if result and result[-1][0] == "en":
                result[-1][1] += part
            else:
                result.append(["en", part])
    print(result)
    return result

def text_to_tensor(text):
    #text = "Hello, [新加坡国立大学], Sound and Music Computing Lab."

    output = split_and_mark_language(text)
    stn_tst_list = []
    lang_ids_list = []
    for item in output:
        language, text = item
        if language == 'zh':
            sep_text, phonemes = get_pinyin_from_text(text)
            # print(phonemes)
            stn_tst = get_text_by_cleaner(phonemes, "chinese_cleaners1", 'Chinese')
            lang_ids_list.append(torch.full((len(stn_tst),), 0, dtype=torch.long))
        else:
            sep_text = text
            phonemes = get_phonemes('English', text)
            tones = phonemes
            stn_tst = get_text_by_cleaner(phonemes, "chinese_cleaners1", 'English')
            lang_ids_list.append(torch.full((len(stn_tst),), 1, dtype=torch.long))

        stn_tst_list.append(stn_tst)

    stn_tst = torch.cat(stn_tst_list, dim=0)
    lang_ids = torch.cat(lang_ids_list, dim=0)
    assert len(lang_ids) == len(stn_tst), "text and language length mismatch, {} != {}".format(len(lang_ids), len(stn_tst))

    return stn_tst, lang_ids


def text_to_tensor2(text):
    output = split_and_mark_language(text)
    stn_tst_list = []
    lang_ids_list = []
    phonemes_list = []
    for item in output:
        language, text = item
        if language == 'zh':
            segs = text.split(' ')
            res = '< '
            for seg in segs:
                initials = to_initials(seg)
                finals = to_finals_tone3(seg, neutral_tone_with_five=True)
                if initials == '':
                    initials = '~'
                res += (initials + ' ' + finals + ' # ')
            phonemes = res[:-2] + '. >'
            stn_tst = get_text_by_cleaner(phonemes, "chinese_cleaners1", 'Chinese')
            lang_ids_list.append(torch.full((len(stn_tst),), 0, dtype=torch.long))
            phonemes_list.append(phonemes)
        else:
            sep_text = text
            phonemes = get_phonemes('English', text)
            stn_tst = get_text_by_cleaner(phonemes, "chinese_cleaners1", 'English')
            lang_ids_list.append(torch.full((len(stn_tst),), 1, dtype=torch.long))
            phonemes_list.append(phonemes)
        stn_tst_list.append(stn_tst)

    stn_tst = torch.cat(stn_tst_list, dim=0)
    lang_ids = torch.cat(lang_ids_list, dim=0)
    assert len(lang_ids) == len(stn_tst), "text and language length mismatch, {} != {}".format(len(lang_ids), len(stn_tst))

    return stn_tst, lang_ids


if __name__ == "__main__":
    text = "You pronounced [gua1] instead of [guang1]"
    stn_tst, lang_ids, phonemes_list = text_to_tensor2(text)
    print(stn_tst)
    print(lang_ids)
    print(phonemes_list)