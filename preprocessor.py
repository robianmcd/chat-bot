import chat_bot.util as util
import pickle
import getopt
import sys

input_files = ['cornell-movie-dialogs-corpus\movie_responses.txt', 'cornell-movie-dialogs-corpus\movie_context.txt']

def build_vocab():
    vocab = util.get_vocab()

    for file_path in input_files:
        with open(file_path, encoding='utf8') as file:
            for i, line in enumerate(file):
                tokens = util.tokenize(line)

                for token in tokens:
                    if not (token in vocab):
                        vocab[token] = len(vocab)

    with open('vocab.pickle', 'wb') as file:
        pickle.dump(vocab, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('{0} words in vocab'.format(len(vocab)))

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'ove', ['overwrite', 'vocab', 'encode'])
    print(opts, args)