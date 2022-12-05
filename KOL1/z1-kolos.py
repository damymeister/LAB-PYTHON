import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv('zadanie.csv')
#ZAD1
#a)
col = data.columns
#b)
val = data.values
#c)
mean_col = np.mean(val, axis=0)
#d)
mean_std = np.std(val)
#e)
diffrence = val - mean_std
#f)
max_row_val = np.max(val, axis=1)
#g)
arr2 = val * 2
#h)
col_numpy = np.array(col)
value_max = col_numpy[np.max(val) == np.max(val,axis=0)]
#i)
arr9 = (val < mean_std).sum(axis=0)


#ZAD2
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