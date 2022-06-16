import numpy as np
import random
from keras.models import Sequential
from keras.layers import Embedding

def createEmbedding(vocab_size, tokenizer, max_length):

    embedding_index = dict()

    # Download this file from https://github.com/stanfordnlp/GloVe -> glove.840B.300d.zip, but don't push it into github.
    f = open('../Datasets/glove.840B.300d.txt', encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]

        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            for pos in range(len(values[1:])):
                if pos == '.' or pos == 'name@domain.com':
                    values[pos + 1] = random.randint(-1, 1)
                    coefs = np.asarray(values[1:], dtype='float32')

        embedding_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embedding_index))

    embedding_matrix = np.zeros((vocab_size, 300))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))

    model.save("EmbeddingLayer.h5")
