import chat_bot.util as util
import pickle
import getopt
import sys
import os

min_word_count_threshold = 5
max_tokens_in_sequence = 200

def build_vocab(input_files):
    word_count_dict = {}

    for file_path in input_files:
        with open(file_path, encoding='utf8') as file:
            for i, line in enumerate(file):
                tokens = util.tokenize(line)

                for token in tokens:
                    if not (token in word_count_dict):
                        word_count_dict[token] = 0
                    word_count_dict[token] = word_count_dict[token] + 1

    vocab = util.get_new_vocab()

    for word, count in word_count_dict.items():
        if count > min_word_count_threshold:
            vocab[word] = len(vocab)


    with open('data/vocab.pickle', 'wb') as file:
        pickle.dump(vocab, file, protocol=pickle.HIGHEST_PROTOCOL)
        print('{0} words in vocab'.format(len(vocab)))

def encode_data(input_files):
    vocab = util.get_vocab()

    for file_path in input_files:
        file_name, ext = os.path.splitext(file_path)

        with open(file_path, encoding='utf8') as input_file:
            output_file_path = file_name + '.encoded' + ext
            with open(output_file_path, 'w') as output_file:
                max_tokens = 0
                for i, line in enumerate(input_file):
                    max_tokens = max(max_tokens, len(util.tokenize(line)))
                input_file.seek(0)

                max_tokens = min(max_tokens_in_sequence, max_tokens)

                for i, line in enumerate(input_file):
                    sequence = util.tokenize(line)[:max_tokens_in_sequence]
                    encoded_tokens = [str(vocab.get(token, vocab['UNKNOWN_WORD'])) for token in sequence]
                    encoded_tokens += ['0'] * (max_tokens - len(encoded_tokens))
                    output_file.write(' '.join(encoded_tokens) + '\n')
                print('Encoded "{0}" to "{1}".'.format(file_path, output_file_path))

#Sample usage: python .\preprocessor.py -d movies-large -v -e
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'd:ve', ['directory=', 'vocab', 'encode'])

    directory = None
    run_build_vocab = False
    run_encode = False

    for opt, arg in opts:
        if opt in ('-d', '--directory'):
            directory = arg
        elif opt in ('-v', '--vocab'):
            run_build_vocab = True
        elif opt in ('-e', '--encode'):
            run_encode = True

    if directory is None:
        sys.exit('You must specify a directory in the data folder to load the data from')

    if run_build_vocab:
        build_vocab([
            'data/{0}/data_context.txt'.format(directory),
            'data/{0}/data_responses.txt'.format(directory)
        ])

    if run_encode:
        encode_data([
            'data/{0}/data_context.txt'.format(directory),
            'data/{0}/data_responses.txt'.format(directory)
        ])