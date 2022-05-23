import csv
import random

import pandas as pd
import keras_preprocessing
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, CuDNNLSTM, SimpleRNN, BatchNormalization, GRU, Dropout
import string
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


def vectorizeSequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results

def accPlot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'ro', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Accuracy of training and validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree


data = pd.read_csv("../Datasets/SpamCSV.csv", encoding="ISO-8859-1")
pd.set_option('display.max_colwidth', None)
data = data[['sms', 'sentiment']]

X = data.drop(columns=['sms']).copy()
y = data['sms']

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)

y_train = y_train.apply(lambda x: remove_punctuation(x))
y_train = y_train.apply(lambda x: x.lower())

max_length = len(max(y_train, key=lambda coll: len(coll)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(y_train)
vocab_size = len(tokenizer.word_index) + 1

y_train = tokenizer.texts_to_sequences(y_train)
y_train = pad_sequences(y_train, maxlen=max_length, padding='post')

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

# ---------- MODEL ------------

model = Sequential()
# model.add(Embedding(vocab_size,300))
# # model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))
# model.add(SimpleRNN(32))
# model.add(BatchNormalization())
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=True))
model.add(GRU(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(y_train, X_train, validation_split=0.2, batch_size=64, shuffle=True, epochs=7)

accPlot(history)

model.save('MaksModel.h5')
