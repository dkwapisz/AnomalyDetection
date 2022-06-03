import csv
import random

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, CuDNNLSTM, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

all_X = []
all_y = []
allSentences = []

with open('../Datasets/SpamCSV.csv', 'r', newline='', encoding='utf8') as csvfile:
    smsReader = csv.DictReader(csvfile)
    for row in smsReader:
        sentence = str(row['sms']).split(sep=" ")
        all_X.append(row['sms'])
        all_y.append(row['sentiment'])
        allSentences.append(sentence)

for i in range(0, len(all_y)):
    all_y[i] = np.array(int(all_y[i]))

all_y = np.array(all_y)

X_train, X_rem, y_train, y_rem = train_test_split(all_X, all_y, train_size=0.6)
X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, train_size=0.5)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_X)
vocab_size = len(tokenizer.word_index) + 1

encoded_sms_train = tokenizer.texts_to_sequences(X_train)
encoded_sms_test = tokenizer.texts_to_sequences(X_test)
encoded_sms_val = tokenizer.texts_to_sequences(X_val)

max_length = len(max(allSentences, key=lambda coll: len(coll)))  # max word length of sms

allSentence = []

padded_sms_train = pad_sequences(encoded_sms_train, maxlen=max_length, padding='post')
padded_sms_test = pad_sequences(encoded_sms_test, maxlen=max_length, padding='post')
padded_sms_val = pad_sequences(encoded_sms_val, maxlen=max_length, padding='post')

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
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sms_train, y_train, epochs=5, validation_data=(padded_sms_val, y_val))

model.save('Model3_LSTM.h5')

model.evaluate(padded_sms_test, y_test)
