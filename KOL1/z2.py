import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
data = pd.read_excel('ex2.xlsx')

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
korr = data.corr()

fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], y)

def testuj(n):
    s = 0 
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 221, shuffle = True )
        LinReg = LinearRegression()
        LinReg.fit(X_train, y_train)
        y_predict = LinReg.predict(X_test)
        s += mean_absolute_error(y_test, y_predict)
    return s/n

def usun(n):
    s=0
    for i in range(n):
        X_train, X_Test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 221, shuffle = True )
        outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
        X_train_no_outliers = X_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        LinReg=LinearRegression()
        LinReg.fit(X_train_no_outliers, y_train_no_outliers)
        y_predict = LinReg.predict(X_Test)
        s+= mean_absolute_error(y_test, y_predict)
    return s/n

def zastap(n):
    s = 0
    for i in range(n):
         X_train, X_Test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 221, shuffle = True )
         outliers = np.abs((y_train - y_train.mean())/ y_train.std())>3
         y_train_mean = y_train.copy()
         y_train_mean[outliers] = y_train.mean()
         LinReg = LinearRegression()
         LinReg.fit(X_train, y_train_mean)
         y_predict = LinReg.predict(X_Test)
         s+= mean_absolute_error(y_test, y_predict)
    return s/n
        









