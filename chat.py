################################################
############## inference models ################
################################################
import numpy as np
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding
import pickle
import chat_bot.util as util
import math

with open('network_config.pickle', 'rb') as file:
    config = pickle.load(file)

#Encoder
encoder_input_layer = Input(shape=(None,))
encoder_embedding_layer = Embedding(config['vocab_size'], config['thought_vector_size'])
encoder_gru_layer = GRU(config['thought_vector_size'], return_state=True)

encoder = encoder_embedding_layer(encoder_input_layer)
encoder, encoder_state = encoder_gru_layer(encoder)

encoder_model = Model([encoder_input_layer], encoder_state)
encoder_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
encoder_embedding_layer.set_weights(config['weights']['encoder_embedding'])
encoder_gru_layer.set_weights(config['weights']['encoder_gru'])

#Decoder
decoder_input_layer = Input(shape=(None,))
decoder_thought_vector_input_layer = Input(shape=(config['thought_vector_size'],))
decoder_embedding_layer = Embedding(config['vocab_size'], config['thought_vector_size'])
decoder_gru_layer = GRU(config['thought_vector_size'], return_sequences=True, return_state=True)
decoder_dense_layer = Dense(config['vocab_size'], activation='softmax')

decoder = decoder_embedding_layer(decoder_input_layer)
decoder, decoder_state = decoder_gru_layer(decoder, initial_state=decoder_thought_vector_input_layer)
decoder = decoder_dense_layer(decoder)

decoder_model = Model([decoder_input_layer, decoder_thought_vector_input_layer], [decoder, decoder_state])
decoder_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
decoder_embedding_layer.set_weights(config['weights']['decoder_embedding'])
decoder_gru_layer.set_weights(config['weights']['decoder_gru'])
decoder_dense_layer.set_weights(config['weights']['decoder_dense'])

print('finished creating models')

vocab = util.get_vocab()
vocab_list = list(vocab.keys())
EOS_label = vocab['END_OF_STRING']

def reply(input_str):
    tokens = util.tokenize(input_str)
    encoded_tokens = [vocab.get(token, vocab['UNKNOWN_WORD']) for token in util.tokenize(input_str)]
    encoded_tokens += [0] * (config['sequence_length'] - len(encoded_tokens))

    # Encode the input as state vectors.
    decoder_state = encoder_model.predict(np.array([encoded_tokens]))

    response_seq = beam_search(5, decoder_state)
    if response_seq[-1] == EOS_label:
        response_seq = response_seq[:-1]
    return ' '.join([vocab_list[label] for label in response_seq])

    # # Populate the first character of target sequence with "end of string". Should maybe use something else for this
    # last_token = np.array([[EOS_label]])

    # stop_condition = False
    # decoded_sentence = ''
    #
    # while not stop_condition:
    #     predicted_token_one_hot, decoder_state = decoder_model.predict([last_token, decoder_state])
    #
    #     # Sample a token
    #     predicted_token_index = np.argmax(predicted_token_one_hot[0, 0])
    #     predicted_word = vocab_list[predicted_token_index]
    #
    #     # Exit condition: either hit max length
    #     # or find stop character.
    #     if predicted_word == 'END_OF_STRING':
    #         stop_condition = True
    #     else:
    #         decoded_sentence += ' ' + predicted_word
    #
    #     if len(decoded_sentence) >= config['sequence_length']:
    #         stop_condition = True
    #
    #     last_token = np.array([[predicted_token_index]])
    #
    # return decoded_sentence

def beam_search(k, initial_state):
    sequences = [{'seq': [EOS_label], 'state': initial_state, 'score': 1}]

    for i in range(config['sequence_length']):
        next_sequences = []
        for seq_i in range(len(sequences)):
            cur_sequence = sequences[seq_i]
            last_token = np.array([[ cur_sequence['seq'][-1] ]])

            #if we have reached the end of a sequence stop predicting new characters
            if last_token == EOS_label and len(cur_sequence['seq']) > 1:
                next_sequences.append(cur_sequence)
            else:
                predicted_token_one_hot, decoder_state = decoder_model.predict([last_token, cur_sequence['state']])

                #get max indicies
                top_k_labels = predicted_token_one_hot[0,0].argsort()[-k:]

                for label in top_k_labels:
                    new_seq = np.concatenate((cur_sequence['seq'], [label]))
                    new_score = cur_sequence['score'] * predicted_token_one_hot[0,0,label]
                    next_sequences.append({'seq': new_seq, 'state': decoder_state, 'score': new_score})
        #keep best k sequences (highest scores)
        sequences = sorted(next_sequences, key=lambda s: s['score'])[-k:]

    #keep top sequence
    return max(sequences, key=lambda s: s['score'])['seq'][1:]


while True:
    msg = input('Message: ')
    print(reply(msg))
