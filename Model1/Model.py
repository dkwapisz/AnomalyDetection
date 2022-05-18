import csv
import random

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, CuDNNLSTM

allSms = []
allLabels = []
allSentences = []

with open('../Datasets/SpamCSV.csv', 'r', newline='', encoding='utf8') as csvfile:
    smsReader = csv.DictReader(csvfile)
    for row in smsReader:
        sentence = str(row['sms']).split(sep=" ")
        allSms.append(row['sms'])
        allLabels.append(row['sentiment'])
        allSentences.append(sentence)

for i in range(0, len(allLabels)):
    allLabels[i] = np.array(int(allLabels[i]))

allLabels = np.array(allLabels)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(allSms)
vocab_size = len(tokenizer.word_index) + 1

encoded_sms = tokenizer.texts_to_sequences(allSms)

max_length = len(max(allSentences, key=lambda coll: len(coll)))  # max word length of sms

allSentence = []

padded_sms = pad_sequences(encoded_sms, maxlen=max_length, padding='post')

embedding_index = dict()

# Download this file from https://github.com/stanfordnlp/GloVe -> glove.840B.300d.zip, but don't push it into github.
f = open('glove.840B.300d.txt', encoding='utf8')
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
#model.add(CuDNNLSTM(300))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sms, allLabels, epochs=5)

model.save('Model1_withoutLSTM.h5')

