""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols_zh, symbols_en


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id_zh = {s: i for i, s in enumerate(symbols_zh)}
_id_to_symbol_zh = {i: s for i, s in enumerate(symbols_zh)}

start_index = len(symbols_zh)
_symbol_to_id_en = {s: i for i, s in enumerate(symbols_en, start=start_index)}
_id_to_symbol_en = {i: s for i, s in enumerate(symbols_en, start=start_index)}

from collections import defaultdict
_symbol_to_id = defaultdict(list)

for i, s in enumerate(symbols_zh):
    _symbol_to_id[s].append(i)

start_index = len(symbols_zh)
for i, s in enumerate(symbols_en, start=start_index):
    _symbol_to_id[s].append(i)


_id_to_symbol = {}
for i, s in enumerate(symbols_zh):
    _id_to_symbol[i] = s

start_index = len(symbols_zh)
for i, s in enumerate(symbols_en, start=start_index):
    _id_to_symbol[i] = s



def text_to_sequence(text, cleaner_names, language, phoneme=False):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  if not phoneme:
    clean_text = _clean_text(text, cleaner_names)
  else:
    clean_text = text

  if language == 'Chinese':
    clean_text = clean_text.split()
  else:
    clean_text = clean_text

  for symbol in clean_text:
    if language == 'Chinese':
      try:
        symbol_id = _symbol_to_id[symbol][0]
      except:
        print(clean_text)
    else:
      symbol_id = _symbol_to_id[symbol][-1]
    sequence += [symbol_id]

  # print(clean_text, sequence, language)
  return sequence

def cleaned_text_to_sequence(cleaned_text, language):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  # if language == 'chinese':
  #   cleaned_text = cleaned_text.split()
  #   sequence = [_symbol_to_id[symbol][0] for symbol in cleaned_text]
  # else:
  #   sequence = [_symbol_to_id[symbol][-1] for symbol in cleaned_text]
  if language == 'chinese':
    cleaned_text = cleaned_text.split()
    sequence = []
    for idx, symbol in enumerate(cleaned_text):
        try:
            value = _symbol_to_id[symbol]
            if not isinstance(value, list) or len(value) == 0:
                raise IndexError(f"⚠️ symbol='{symbol}': {value}")
            sequence.append(value[0])
        except Exception as e:
            print(e)
  else:
    sequence = []
    for idx, symbol in enumerate(cleaned_text):
        try:
            value = _symbol_to_id[symbol]
            if not isinstance(value, list) or len(value) == 0:
                raise IndexError(f"symbol='{symbol}' : {value}")
            sequence.append(value[-1])
        except Exception as e:
            print(e)

  return sequence


def sequence_to_text(sequence, language):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
