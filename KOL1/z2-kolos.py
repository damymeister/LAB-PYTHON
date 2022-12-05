import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv('zadanie.csv')
X = data.iloc[:,:data.shape[1]-1]
y = data.iloc[:,-1]

macierz_korelacji = data.corr()

fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], y)

def test(n):
    x=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 221, shuffle = True )
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_predict = linReg.predict(X_test)
        x += mean_absolute_error(y_test, y_predict)
    return x/n

test(10)
