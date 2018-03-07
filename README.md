# chat-bot

## Experiments / Todos:
 - share the embedding layer between encoder and decoder
 - graph loss vs. validation loss
 - use a pretrained vocabulary/embedding vectors
     - get glov vector for each word in vocab
     - feed vectors into embedding layer weights
     - use embedding layer or just send word vectors directly into GRU layer
 - use dropout
 - more epochs
 - Deeper network
 - try outputting embedding vector
     - Either use glov and disable training or train embedding seperately from the rest of the network.
     - would need find the closest word vector: https://stackoverflow.com/questions/32446703/find-closest-vector-from-a-list-of-vectors-python
 - try outputting encoded characters
 - add special tokens for: start of string, and unknown token
 - Run on GPU (oculus machine?)
 - gracefully handel words not in the vocab
 - try lstm
 - use characters instead of words

## Resources
 - https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
 - https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
 - http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 - https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras