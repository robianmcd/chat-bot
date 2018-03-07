import numpy as np
from keras.models import Model
from keras.layers import Input, GRU, Dense, Embedding
from keras.utils import to_categorical
import pickle
import chat_bot.util as util
import getopt
import sys

# Run on CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

vocab = util.get_vocab()

opts, args = getopt.getopt(sys.argv[1:], 'd:', ['directory='])

with open('data/glove_embeddings.pickle', 'rb') as handle:
    embedding_matrix = pickle.load(handle)

directory = None

for opt, arg in opts:
    if opt in ('-d', '--directory'):
        directory = arg

if directory is None:
    sys.exit('You must specify a directory in the data folder to load the data from')

model_input_file = 'data\{0}\data_context.encoded.txt'.format(directory)
model_output_file = 'data\{0}\data_responses.encoded.txt'.format(directory)
THOUGHT_VECTOR_SIZE = 100

encoder_input_data = np.loadtxt(model_input_file)
decoder_target_data = np.loadtxt(model_output_file)
decoder_input_data = decoder_target_data[:, :-1]
decoder_input_data = np.insert(decoder_input_data, 0, values=0, axis=1)
#decoder_target_one_hot = to_categorical(decoder_target_data, len(vocab))

def generateData(batch_size):
    for i in range(len(decoder_input_data)):
        for j in range(batch_size):
            encoder_input_batch = encoder_input_data[i:i+batch_size]
            decoder_input_batch = decoder_input_data[i:i+batch_size]
            decoder_target_one_hot_batch = to_categorical(decoder_target_data[i:i+batch_size], len(vocab))
            #print(encoder_input_data.shape, decoder_input_batch.shape, decoder_target_one_hot_batch.shape)
            yield ([encoder_input_batch, decoder_input_batch], decoder_target_one_hot_batch)


#Create layers
encoder_input_layer = Input(shape=(None,))
encoder_embedding_layer = Embedding(len(vocab), THOUGHT_VECTOR_SIZE, weights=[embedding_matrix])
encoder_gru_layer = GRU(THOUGHT_VECTOR_SIZE, return_state=True)

decoder_input_layer = Input(shape=(None,))
decoder_embedding_layer = Embedding(len(vocab), THOUGHT_VECTOR_SIZE, weights=[embedding_matrix])
decoder_gru_layer = GRU(THOUGHT_VECTOR_SIZE, return_sequences=True)
decoder_dense_layer = Dense(len(vocab), activation='softmax')


#connect network
encoder = encoder_embedding_layer(encoder_input_layer)
encoder, encoder_state = encoder_gru_layer(encoder)

decoder = decoder_embedding_layer(decoder_input_layer)
decoder = decoder_gru_layer(decoder, initial_state=encoder_state)
decoder = decoder_dense_layer(decoder)

#small batch size works well on CPU e.g. 10. large on GPU
#however larger batch size will use up RAM
batch_size = 30
model = Model([encoder_input_layer, decoder_input_layer], decoder)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit_generator(generateData(batch_size=batch_size), steps_per_epoch=len(decoder_input_data)//batch_size, epochs=3)

# model.fit([encoder_input_data, decoder_input_data], decoder_target_one_hot,
#           batch_size=32,
#           epochs=10,
#           validation_split=0.2)

network_config = {
    'vocab_size': len(vocab),
    'thought_vector_size': THOUGHT_VECTOR_SIZE,
    'sequence_length': encoder_input_data.shape[1],
    'weights': {
        'encoder_embedding': encoder_embedding_layer.get_weights(),
        'encoder_gru': encoder_gru_layer.get_weights(),
        'decoder_embedding': decoder_embedding_layer.get_weights(),
        'decoder_gru': decoder_gru_layer.get_weights(),
        'decoder_dense': decoder_dense_layer.get_weights()
    }
}

with open('network_config.pickle', 'wb') as file:
    pickle.dump(network_config, file)

print('saved network config to "{}". Vocab size: {}. Thought vector size: {}. Sequence length: {}.'
    .format(
    'network_config.pickle',
    network_config['vocab_size'],
    network_config['thought_vector_size'],
    network_config['sequence_length']
)
)