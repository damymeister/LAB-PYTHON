# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:41:55 2022

@author: Alicja
"""

from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

#5.2
data = load_digits()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y)

plt.gray()
plt.matshow(data.images[515])


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_val = X_train[:400]
y_val = y_train[:400]
X_train = X_train[401:]
y_train = y_train[401:]

model = Sequential()
model.add(layers.Dense(64, 
                       input_shape = (X_train.shape[1],), 
                       activation = 'relu' ))
model.add(layers.Dense(64, activation=('relu')))
model.add(layers.Dense(y_train.shape[1], activation = 'softmax'))

model.summary()

model.compile(optimizer= 'Adam',
              loss = 'categorical_crossentropy',
              metrics = 'acc')

history = model.fit(X_train, y_train, batch_size = 8, epochs=5, validation_data = (X_val, y_val),
                    verbose = 1)

y_pred = model.predict(X_test)


floss_train = history.history['loss']
floss_test = history.history['val_loss']
acc_train = history.history['acc']
acc_test = history.history['val_acc']
fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 5)
ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()

#%%
#6.1
from keras.regularizers import l2, l1
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout, GaussianNoise
from keras.layers import LayerNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from keras.regularizers import l2, l1
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values

y = pd.Categorical(y)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.2)

neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001

block = [Dense,]

args = [(neuron_num,'selu'),(),(),(do_rate,),(noise,)]

reg_rate = [0, 0.0001, 0.001, 0.01, 0.1]
for rate in reg_rate:
    model = Sequential()
    model.add(Dense(neuron_num, activation='relu',input_shape = (X.shape[1],),kernel_regularizer=l2(rate)))
    for i in range(2):
        for layer,arg in zip(block, args):
            model.add(layer(*arg))
    
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(optimizer= Adam(learning_rate),loss='binary_crossentropy',metrics=('accuracy', 'Recall', 'Precision'))
    
    model.fit(X_train, y_train, batch_size=32,epochs=5, validation_data=(X_test, y_test),verbose=1)
    acc=max(model.history.history['val_accuracy'])


#%%
#6.2
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values

y = pd.Categorical(y)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

neuron_num = 64
noises = [0, 0.1, 0.2, 0.3]#noises dotyczy gaussian noise (block)
learning_rate = 0.001

for noise in noises:
    block = [
        Dense,
        GaussianNoise,
        ]
    
    args = [(neuron_num,'selu'),(noise,)]
    
    model = Sequential()
    model.add(Dense(neuron_num, activation='relu',input_shape = (X.shape[1],)))
    for i in range(2):
        for layer,arg in zip(block, args):
            model.add(layer(*arg))
        
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(optimizer= Adam(learning_rate),loss='binary_crossentropy',metrics=('accuracy', 'Recall', 'Precision'))
        
    model.fit(X_train, y_train, batch_size=32,epochs=5, validation_data=(X_test, y_test),verbose=1)
    
acc=max(model.history.history['val_accuracy'])
    
#6.3
data = load_iris()
y = data.target
X = data.data
y = pd.Categorical(y)
y_one_hot = pd.get_dummies(y).values

y = pd.Categorical(y)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

neuron_num = 64
noises = [0, 0.1, 0.2, 0.3]
learning_rate = 0.001
for noise in noises:
    block = [Dense,GaussianNoise]
    
    
    args = [(neuron_num,'selu'),(noise,)]
     
    model = Sequential()
    model.add(Dense(neuron_num, activation='relu', input_shape = (X.shape[1],),kernel_regularizer = l2(0.01)))
        
    repeat_num = 2
        
    for i in range(repeat_num):
        for layer,arg in zip(block, args):
            model.add(layer(*arg))
        
    model.add(Dense(y_train.shape[1], activation = 'softmax'))
    model.compile(optimizer= Adam(learning_rate),loss='binary_crossentropy',metrics=('accuracy', 'Recall', 'Precision'))
        
    history = model.fit(X_train, y_train, batch_size=32,epochs=5, validation_data=(X_test, y_test),verbose=1)
        
acc=max(model.history.history['val_accuracy'])

