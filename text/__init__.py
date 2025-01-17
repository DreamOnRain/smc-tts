""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols_zh, symbols_en


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id_en = {s: i for i, s in enumerate(symbols_en)}
_id_to_symbol_en = {i: s for i, s in enumerate(symbols_en)}
_symbol_to_id_zh = {s: i for i, s in enumerate(symbols_zh)}
_id_to_symbol_zh = {i: s for i, s in enumerate(symbols_zh)}

def text_to_sequence(text, cleaner_names, language, phoneme=False):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  if language == 'Chinese':
    _symbol_to_id = _symbol_to_id_zh
  else:
    _symbol_to_id = _symbol_to_id_en

  if not phoneme:
    clean_text = _clean_text(text, cleaner_names)
  else:
    clean_text = text

  if language == 'Chinese':
    clean_text = clean_text.split()
  else:
    clean_text = clean_text

  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text, language):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  if language == 'Chinese':
    _symbol_to_id = _symbol_to_id_zh
    cleaned_text = cleaned_text.split()
  else:
    _symbol_to_id = _symbol_to_id_en
  # sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence


def sequence_to_text(sequence, language):
  '''Converts a sequence of IDs back to a string'''
  if language == 'Chinese':
    _id_to_symbol = _id_to_symbol_zh
  else:
    _id_to_symbol = _id_to_symbol_en
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
