from sklearn.datasets import load_digits
import pandas as pd 
import numpy as np
data = load_digits()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_val = X_train[:400]
y_val = y_train[:400]

X_train = X_train[400:]
y_train = y_train[400:]

model = Sequential()
model.add(layers.Dense(64, input_shape = (X_train.shape[1],), activation = 'relu'))
model.add(layers.Dense(64, activation ='relu'))




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