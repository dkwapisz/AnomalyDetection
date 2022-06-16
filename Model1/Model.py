import csv
import random

import keras.models

from CreateEmbeddingLayer import createEmbedding
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

# This creates embedding
# createEmbedding(vocab_size=vocab_size, tokenizer=tokenizer, max_length=max_length)

# ---------- MODEL ------------
embeddingLayer = keras.models.load_model("EmbeddingLayer.h5")

model = Sequential()
model.add(embeddingLayer)
model.add(Bidirectional(CuDNNLSTM(512, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sms_train, y_train, epochs=5, validation_data=(padded_sms_val, y_val))

model.save('NextModel.h5')

model.evaluate(padded_sms_test, y_test)
