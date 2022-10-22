
from statistics import LinearRegression
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

plik = pd.read_excel("practice_lab_2.xlsx")

korelacje = plik.corr()

X = plik.iloc[:,:plik.shape[1]-1]#Nie bierzemy ostatniej kolumny
y = plik.iloc[:,-1]

#ZAD#2-1 
#nw czego nie dziala fig, ax = plt.subplot(X.shape[1],1,figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col],y)
#ZAD2-2
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def testuj_model(n):
    s=0 #suma
    for i in range(n):
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0)
linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
s+= mean_absolute_percentage_error(y_test, y_pred)
return s/n
testuj_model(10)
#ZAD2-3
def usuniecie(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        X_train_no_outliers = X_train[~outliers, :]
        y_train_no_outliers = y_train[~outliers]
        linReg = LinearRegression(X_train_no_outliers,y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        s+= mean_absolute_percentage_error (y_test, y_pred)
        return s/n
usuniecie(10)

def zamiana(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        X_train_no_outliers = X_train[~outliers, :]
        y_train_no_outliers = y_train[~outliers]
        linReg = LinearRegression(X_train_no_outliers,y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        s+= mean_absolute_percentage_error (y_test, y_pred)
        return s/n


X_train, X_test, y_train, y_test = 
train_test_split(X,y, test_size=0.2, random_state=221, shuffle=True)
linReg = LinearRegression()
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
