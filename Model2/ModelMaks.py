import csv
import random

import pandas as pd
import numpy as np
import Utils.Utils as Utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, CuDNNLSTM, SimpleRNN, BatchNormalization, GRU, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

data = pd.read_csv("../Datasets/SpamCSV.csv", encoding="ISO-8859-1")
pd.set_option('display.max_colwidth', None)
data = data[['sms', 'sentiment']]

y = data.drop(columns=['sms']).copy()
X = data['sms']

print(X)
print(y)
y_train, y_rem, X_train, X_rem = train_test_split(y, X, train_size=0.7)

X_train = X_train.apply(lambda x: Utils.remove_punctuation(x))
X_train = X_train.apply(lambda x: x.lower())

max_length = len(max(X_train, key=lambda coll: len(coll)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')

embedding_index = dict()

# Download this file from https://github.com/stanfordnlp/GloVe -> glove.840B.300d.zip, but don't push it into github.
f = open('../Model1/glove.840B.300d.txt', encoding='utf8')

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
history = model.fit(x=X_train, y=y_train, validation_split=0.2, batch_size=32, shuffle=True, epochs=4)

print(vocab_size)
Utils.accPlot(history)
Utils.lossPlot(history)

X_rem = X_rem.apply(lambda x: Utils.remove_punctuation(x))
X_rem = X_rem.apply(lambda x: x.lower())

max_length = len(max(X_rem, key=lambda coll: len(coll)))

# tokenizer = Tokenizer()
# vocab_size = len(tokenizer.word_index) + 1
X_rem = tokenizer.texts_to_sequences(X_rem)
X_rem = pad_sequences(X_rem, maxlen=max_length, padding='post')
response = model.evaluate(x=X_rem, y=y_rem)

model.save('MaksModel.h5')
