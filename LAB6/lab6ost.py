#%%ZAD 6.1
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import  Dense, GaussianNoise, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001

block = [
    Dense,
    ]
args = [(neuron_num,'selu'),(),(),(do_rate,),(noise,)]

model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_shape = (X.shape[1],)))


reg_rate = [0, 0.0001, 0.001, 0.01, 0.1]
mean_acc = []

for rate in reg_rate:
    model = Sequential()
    model.add(Dense(neuron_num, input_shape=(X.shape[1],), activation='relu', kernel_regularizer=l2(rate)))
    for i in range(2):
        for layer, arg in zip(block, args):
            model.add(layer(*arg))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=('accuracy', 'Recall', 'Precision'))
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), verbose=1)
    mean_acc.append(np.mean(model.history.history['val_accuracy']))
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(reg_rate, mean_acc, label = 'average accuracy')
ax.set_title('Wykres pokazujacy zaleznosc Sredniej dokladnosci od wspolczynnika regularyzacji')
ax.legend()
#%% ZAD 6.2

data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values

y = pd.Categorical(y)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

noise = 0.1
block = [Dense,]
model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_shape = (X.shape[1],)))
do_rate = [0, 0.2, 0.3, 0.5]
mean_acc = []

for rate in do_rate:
    args = [(neuron_num,'selu'),(),(),(rate),(noise,)]
    model = Sequential()
    model.add(Dense(neuron_num, input_shape=(X.shape[1],), activation='relu'))
    for i in range(2):
        for layer, arg in zip(block, args):
            model.add(layer(*arg))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=('accuracy', 'Recall', 'Precision'))
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), verbose=1)
    mean_acc.append(np.mean(model.history.history['val_accuracy']))
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(do_rate, mean_acc, label = 'average accuracy')
ax.set_title('Wykres punktowy pokazujacy zaleznosc sredniej dokladnosci do do_rate')
ax.legend()
#%% ZAD 6.3
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

noises = [0, 0.1, 0.2, 0.3]
block = [Dense, GaussianNoise]
model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_shape = (X.shape[1],)))
mean_acc = []

for noise in noises:
    args = [(neuron_num,'selu'),(noise,)]
    model = Sequential()
    model.add(Dense(neuron_num, input_shape=(X.shape[1],), activation='relu'))
    for i in range(2):
        for layer, arg in zip(block, args):
            model.add(layer(*arg))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=('accuracy', 'Recall', 'Precision'))
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), verbose=1)
    mean_acc.append(np.mean(model.history.history['val_accuracy']))
fig,ax = plt.subplots(1,1, figsize=(10,10))
ax.scatter(noises, mean_acc, label = 'average accuracy')
ax.set_title('Wykres punktowy pokazujacy zaleznosc sredniej dokladnosci do parametru noise')
ax.legend()
#%%