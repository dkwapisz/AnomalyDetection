from keras import Sequential, layers
import LoadData
import numpy as np
import Plots

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
model.add(layers.Dense(16, activation='relu', input_shape=(length,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val))

model.evaluate(x_test, y_test)

Plots.accPlot(history)
Plots.lossPlot(history)

