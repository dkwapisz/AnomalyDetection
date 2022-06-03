from random import random

import numpy as np
import pandas as pd
from keras import Input
from keras import layers
from keras import Model
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from Utils import Utils

false_data = pd.read_csv("../Datasets/Fake.csv", encoding="ISO-8859-1")
false_data["Label"] = 0

true_data = pd.read_csv("../Datasets/True.csv", encoding="ISO-8859-1")
true_data["Label"] = 1

final_data = pd.concat([true_data, false_data])
final_data = shuffle(final_data)

print(final_data.shape)

pd.set_option('display.max_colwidth', None)
final_data = final_data[['title', 'text', 'Label']]

y = final_data['Label']

# ------- EMBEDDING --------
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

# ----------- TITLE -------------
title = final_data['title']
title = title.apply(lambda x: Utils.remove_punctuation(x))
title = title.apply(lambda x: x.lower())

max_length_title = len(max(title, key=lambda coll: len(coll)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(title)
vocab_size_title = len(tokenizer.word_index) + 1

title = tokenizer.texts_to_sequences(title)
title = pad_sequences(title, maxlen=max_length_title, padding='post')

# -------TITLE EMBEDDING MATRIX --------
title_embedding_matrix = np.zeros((vocab_size_title, 300))

for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        title_embedding_matrix[i] = embedding_vector

# ----------- BODY -------------
body = final_data['text']
body = body.apply(lambda x: Utils.remove_punctuation(x))
body = body.apply(lambda x: x.lower())

max_length_body = len(max(body, key=lambda coll: len(coll)))

tokenizer.fit_on_texts(body)
vocab_size_body = len(tokenizer.word_index) + 1

body = tokenizer.texts_to_sequences(body)
body = pad_sequences(body, maxlen=max_length_body, padding='post')

# -------BODY EMBEDDING MATRIX --------
body_embedding_matrix = np.zeros((vocab_size_body, 300))

for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        body_embedding_matrix[i] = embedding_vector

# ---------- MODEL FUNCTIONAL API------------

title_input = Input(shape=(max_length_title,), dtype='float32', name='title')

title_embedded = layers.Embedding(vocab_size_title, 300, weights=[title_embedding_matrix],
                                  input_length=max_length_title)(title_input)

title_encoded = layers.GRU(16)(title_embedded)

body_input = Input(shape=(max_length_body,), dtype='float32', name='body')

body_embedded = layers.Embedding(vocab_size_body, 300, weights=[body_embedding_matrix],
                                 input_length=max_length_body)(body_input)

body_encoded = layers.GRU(32)(body_embedded)

concatenated = layers.concatenate([title_encoded, body_encoded], axis=-1)

answers = layers.Dense(1, activation='softmax')(concatenated)

model = Model([title_input, body_input], answers)

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit({'title': title, 'body': body}, y, validation_split=0.2, batch_size=128, shuffle=True, epochs=10)

Utils.accPlot(history)
Utils.lossPlot(history)
