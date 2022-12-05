#%%
#7.2
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

train, test = mnist.load_data()#ladujemy zbior

X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]

X_train = np.expand_dims(X_train, axis=-1)#expand dims rozszerza ksztalt naszej tablicy
X_test = np.expand_dims(X_test, axis=-1)

class_cnt = np.unique(y_train).shape[0]#z naszeego y traina wybieramy unikatowe wartosci

filter_cnt = 32#liczba filtrow, ekstraktuje cechy i przekazuje je dalej
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'#funkcja aktywacji
kernel_size = (3,3)#rozmiar lat
model = Sequential()#tworzymy model sekwencyjny
conv_rule = 'same'#zeby obraz wejsciowy byl rowny wyjsciowemu  ( dopelnia zerami )
model.add(Conv2D(input_shape = X_train.shape[1:],filters=filter_cnt,kernel_size = kernel_size, padding = conv_rule, activation = act_func))
model.add(Flatten())#warstwa przeksztalca tab dwuwymiarowa na jednowymiarowa
model.add(Dense(class_cnt, activation='softmax'))#zawsze po conv musi byc flatten a na koncu DENSE
model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy', metrics='accuracy')
model.summary()#Podsumowanie modelu
model.fit(x = X_train, y = y_train, epochs = class_cnt , validation_data=(X_test, y_test))#jezeli nie bedzie nic podane to robiy tak i robimy predict
#%%
#7.3
filter_cnt = 32#liczba filtrow, ekstraktuje cechy i przekazuje je dalej
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'#funkcja aktywacji
kernel_size = (3,3)#rozmiar lat
pooling_size = (2,2)#DZIELI NA POLA DWUWYMIAROWE (przechodzi ramka po zdjeciu i wyciaga maksmalne wartosci z kazdej ramki zeby zmniejszych obraz o polowe)
model = Sequential()#tworzymy model sekwencyjny
conv_rule = 'same'
model.add(Conv2D(input_shape = X_train.shape[1:],filters=filter_cnt,kernel_size = kernel_size, padding = conv_rule, activation = act_func))
model.add(MaxPooling2D(pooling_size))
model.add(Flatten())#warstwa przeksztalca tab dwuwymiarowa na jednowymiarowa
model.add(Dense(class_cnt, activation='softmax'))#zawsze po conv musi byc flatten a na koncu DENSE
model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy', metrics='accuracy')
model.summary()#Podsumowanie modelu
model.fit(x = X_train, y = y_train, epochs = class_cnt , validation_data=(X_test, y_test))#jezeli nie bedzie nic podane to robiy tak i robimy predict

#7.4
#%%
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from keras.datasets import cifar10
(X_train, y_train),(X_test,y_test) = cifar10.load_data()
class_cnt = np.unique(y_train).shape[0]#z naszeego y traina wybieramy unikatowe wartosci
filter_cnt = 32#liczba filtrow, ekstraktuje cechy i przekazuje je dalej
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'#funkcja aktywacji
kernel_size = (3,3)#rozmiar lat
model = Sequential()#tworzymy model sekwencyjny
conv_rule = 'same'#zeby obraz wejsciowy byl rowny wyjsciowemu  ( dopelnia zerami )
model.add(Conv2D(input_shape = X_train.shape[1:],filters=filter_cnt,kernel_size = kernel_size, padding = conv_rule, activation = act_func))
model.add(Flatten())#warstwa przeksztalca tab dwuwymiarowa na jednowymiarowa
model.add(Dense(class_cnt, activation='softmax'))#zawsze po conv musi byc flatten a na koncu DENSE
model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy', metrics='accuracy')
model.summary()#Podsumowanie modelu
model.fit(x = X_train, y = y_train, epochs = class_cnt , validation_data=(X_test, y_test))
#%%