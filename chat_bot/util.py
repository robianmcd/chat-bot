import pickle
from collections import OrderedDict
from nltk.tokenize.casual import TweetTokenizer
import re

def get_vocab():
    try:
        with open('data/vocab.pickle', 'rb') as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        return get_new_vocab()

def get_new_vocab():
    vocab = OrderedDict()
    vocab['end of string'] = 0
    return vocab

tknzr = TweetTokenizer(reduce_len=True)
def tokenize(str):
    str = re.sub(r'(:[\w_]+:)', r'<\1>', str)
    return tknzr.tokenize(str)


