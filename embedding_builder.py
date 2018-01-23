import urllib.request
import zipfile
import os.path
import sys
import chat_bot.util as util
import numpy as np
import pickle

glove_zip_file = 'downloads/glove.twitter.27B.zip'
glove_txt_file = 'downloads/glove.twitter.27B.100d.txt'
word_vector_size = 100

#Taken from https://stackoverflow.com/a/13895723/373655
def report_hook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


if not os.path.isdir('downloads'):
    os.makedirs('downloads')

if not os.path.isfile(glove_zip_file):
    print('Downloading {}'.format(glove_zip_file))
    urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.twitter.27B.zip', glove_zip_file, report_hook)

if not os.path.isfile(glove_txt_file):
    print('Unzipping {}'.format(glove_zip_file))
    with zipfile.ZipFile(glove_zip_file,'r') as zip_ref:
        zip_ref.extractall('downloads')

vocab = util.get_vocab()
embedding_matrix = np.zeros((len(vocab), word_vector_size))
words_found = 0
with open(glove_txt_file, 'r', encoding='utf-8') as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        word_vector = values[1:]
        if word in vocab:
            word_index = vocab[word]
            embedding_matrix[word_index] = np.asarray(word_vector, dtype='float32')

            words_found += 1
            print('{}/{} words found.'.format(words_found, len(vocab)), end='\r')



    print('\nFinished building embedding vector')

vocab_list = list(vocab.items())
missing_words = []
for i, row in enumerate(embedding_matrix):
    if np.array_equal(row, np.zeros(word_vector_size)):
        missing_words.append(vocab_list[i][0])

print('{} words from vocabulary without pretrained GloVe embeddings:'.format(len(missing_words)))

print(missing_words[:100])

with open('data/glove_embeddings.pickle', 'wb') as file:
    pickle.dump(embedding_matrix, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Embedding weights written to data/glove_embeddings.pickle')
