import pickle
from collections import OrderedDict
import keras.preprocessing.text
#from nltk.tokenize.casual import TweetTokenizer

import re

def get_vocab():
    try:
        with open('data/vocab.pickle', 'rb') as handle:
            return pickle.load(handle)
    except FileNotFoundError:
        return get_new_vocab()

def get_new_vocab():
    vocab = OrderedDict()
    vocab['END_OF_STRING'] = 0
    vocab['UNKNOWN_WORD'] = 1
    return vocab

#The twitter tokenizer doesn't split contractions and doesn't match the tokens
#in the pretrained glove embeddings
#tknzr = TweetTokenizer(reduce_len=True, preserve_case=False)

def tokenize(str):
    #Escape slack mentions
    #str = re.sub(r'(:[\w_]+:)', r'<\1>', str)

    #This is a very primitive tokenizer. Would be better to use the Stanford
    #Tokenizer to work with pretrained GloVe embeddings...but it uses Java and
    #is a pain to configure
    # - https://nlp.stanford.edu/software/tokenizer.shtml
    # - https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software
    return keras.preprocessing.text.text_to_word_sequence(str, filters='!"#$%&()*+,-—./:;<=>?@[\\]^_`{|}~\t\n\'')
