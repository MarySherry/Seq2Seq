import numpy as np
import re
import string
import tensorflow_datasets as tfds

def cleaning(text):
    ''' Performs cleaning of text of punctuation, digits,
        excessive spaces and transfers to lower-case
    '''
    exclude = set(string.punctuation + string.digits + '«»…‘’―–—‽')
    text = text.lower().strip()
    text = re.sub(r'\s+', " ", text)
    text = ''.join(character for character in text if character not in exclude)

    return text

def process_data(data, max_sentence_length, SOS, EOS):
    ''' Performs data cleaning, filtering of maximal allowed sentence length, appending of Start-of-String
        and End-of-String characters
    '''
    processed_data = data.copy()
    cleaner = lambda x: cleaning(x)
    processed_data.en = processed_data.en.apply(cleaner)
    processed_data.ru = processed_data.ru.apply(cleaner)

    target_seq_lens = np.array([len(sentence) for sentence in processed_data.ru])
    target_idx = np.where(target_seq_lens <= max_sentence_length)[0]
    keep_idx = target_idx
    print("{} input sentence pairs with {} or fewer words in target language".format(len(keep_idx), max_sentence_length))

    # Append Start-Of-Sequence and End-Of-Sequence tags to target sentences:
    processed_data.ru = processed_data .ru.apply(lambda x: SOS + ' ' + x + ' ' + EOS)

    # Subset initial data
    return processed_data.iloc[keep_idx]

def tokenize_data(data, vocab_size=2**15):
    ''' Performs subword data tokenizing with a given vocabulary size'''
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        data, target_vocab_size=vocab_size)

    return tokenizer