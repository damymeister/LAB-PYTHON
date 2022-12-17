# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:48:34 2022

@author: Alicja
"""
#%%
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merging import concatenate
from keras.layers import Dense, Conv2D, MaxPooling2D,GlobalAveragePooling2D,Input, Flatten
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import pandas as pd

data = mnist.load_data()

X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]

X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)

y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values
filter_cnt = 32
kernel_size = (3,3)
act_func = 'selu'
class_cnt = y_train.shape[1]


#Zadanie 8.1
import tensorflow as tf

def add_inseption_module(input_tensor):
    act_func = "relu"
    paths = [
        [Dense(512, act_func),
         Dense(128, act_func),
         Dense(64, act_func),
         Dense(16, act_func),
         Dense(10, act_func)],
        [Dense(512, act_func),
         Dense(64, act_func),
         Dense(10, act_func),],
        [Dense(512, act_func),
         Dense(64, act_func),
         Dense(10, act_func)
         ],
        [Dense(512, act_func),
         Dense(64, act_func),
         Dense(10, act_func)
         ],
        [Dense(512, act_func),
         Dense(64, act_func),
         Dense(10, act_func)
         ]
        ]
    for_con_tab = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
        for_con_tab.append(output_tensor)
    return for_con_tab    


output_tensor = input_tensor = Input(X_train.shape[1:])
output_tensor = tf.keras.layers.Reshape((784,))(output_tensor)
output_tensor = tf.keras.layers.BatchNormalization()(output_tensor)  
output_tensor = add_inseption_module(output_tensor)
output_tensor = tf.keras.layers.Average()(output_tensor)

ANN = Model(inputs = input_tensor, outputs = output_tensor)
ANN.compile(loss = 'categorical_crossentropy',metrics = 'accuracy', optimizer = 'adam')


from keras.utils.vis_utils import plot_model
plot_model(ANN, show_shapes=True)

#$$
#%%
#8.2

def rest_net(input_tensor):
    skip_tensor = input_tensor
    output_tensor = input_tensor
    output_tensor = Conv2D(32, (3,3), padding='same')(output_tensor)
    output_tensor = tf.keras.layers.BatchNormalization()(output_tensor)
    output_tensor = tf.keras.layers.Add(output_tensor, skip_tensor)(output_tensor)
    output_tensor = tf.keras.layers.Activation('relu')(output_tensor)
    return output_tensor