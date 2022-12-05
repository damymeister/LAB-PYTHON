#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
data = load_diabetes()
plik = pd.DataFrame(data.data, columns = data.feature_names)
plik['target'] = data.target #Wartosc cechy zaleznej podana jest w obiekce data z kluczem target
koleracja = plik.corr() #Tworzymy macierz korelacji
X = plik.iloc[:,:plik.shape[1]-1]#Shape zwraca wymiary tab, shape[1] liczbe kolumn
Y = plik.iloc[:,-1]#Iloc sprawia ze mozna zastosowac zwykle indeksowanie
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20)) #fig-obraz, ax-siatka obrazow, x.shape[1] liczba cech niezaleznych, 1 kolumna wykresow
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], Y)#Dla wybranej cechy niezaleznej(kolumny) wykres punktowy

def test(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=221, shuffle=True)#Losowy podzial na zbior treningowy i testowy        
        linearReg = LinearRegression()
        linearReg.fit(X_train, y_train)#Kalukacja regresji (zbior treningowy)
        y_predict = linearReg.predict(X_test)#Przewidywanie y na podstawie x_testowego
        s += mean_absolute_percentage_error(y_test, y_predict)#Obliczanie wartosci bezwzg. w % miedzy zbiorem testowym a przewidywanym
    return s/n
def usun(n):
    s=0
    for i in range(n):
         X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=221, shuffle=True)
         outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
         X_train_no_outliers = X_train.loc[~outliers,:]#Wszystkie cechy niezalezne bez nietypowych
         Y_train_no_outliers = y_train.loc[~outliers]#Cecha zalezna bez nietypowych
         linearReg = LinearRegression()
         linearReg.fit(X_train_no_outliers, Y_train_no_outliers)
         y_predict = linearReg.predict(X_test)
         s+=mean_absolute_percentage_error(y_test, y_predict)
    return s/n
#Wartosci obserwacji odstajacych zastepujemy srednimi calego zbioru, zmieniamy tylko cechy zalezne
def zastap(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
        y_train_mean = y_train.copy()#Kopiujemy zbior treningowy
        y_train_mean[outliers] = y_train.mean()#Zamiana odstajacych wartosci na usrednione wartosci
        linearReg= LinearRegression()
        linearReg.fit(X_train, y_train_mean )#Liczymy ze zbiorem treningowym usrednionym
        y_predict = linearReg.predict(X_test)
        s+=mean_absolute_percentage_error(y_test, y_predict)
    return s/n
test(9)
usun(9)
zastap(9)
#%%




