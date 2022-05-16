from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from Model_Test import LoadData
import numpy as np
import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.config.list_physical_devices('GPU')))
else:
    print("Please install GPU version of TF")

def vectorizeSequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results


train_data, train_labels, val_data, val_labels, test_data, test_labels, length = LoadData.getData()

x_train = vectorizeSequences(train_data, length)
y_train = np.asarray(train_labels).astype('float32')

x_val = vectorizeSequences(val_data, length)
y_val = np.asarray(val_labels).astype('float32')

x_test = vectorizeSequences(test_data, length)
y_test = np.asarray(test_labels).astype('float32')

model = Sequential()

model.add(Embedding(input_dim=length, output_dim=100))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
