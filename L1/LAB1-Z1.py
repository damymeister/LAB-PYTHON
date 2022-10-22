import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_excel('practice_lab_1.xlsx')
#ZAD-2.1
cols = data.columns
values = data.values
arr1= values[::2,:]
arr2= values[1::2,:]
arr = arr1-arr2
#ZAD-2.2
srednia = values.mean()
odch = values.std()
array3 = (values - srednia ) / odch
#ZAD-2.3
srednia2 = values.mean(axis=0)
odch2 = values.mean(axis=0)
array4 = (values - srednia2) / (odch2+ np.spacing(odch2))
#ZAD-2.4
array5 = (srednia2) / (odch2 + np.spacing(odch2))
#ZAD-2.5
array6 = np.argmax(array5)
#ZAD-2.6
array7 = (values > values.mean(axis=0)).sum(axis=0)
#ZAD-2.7
max_array = values.max()
max_cols = values.max(axis=0)
cols = np.array(cols)
cols[max_array==max_cols]
#ZAD-2.8
maska = values == 0
tab_zero = np.sum(maska, axis=0)
max_tabzero = max(tab_zero)
cols = np.array(cols)
print(cols[tab_zero == max_tabzero])
#ZAD-2.9
suma_parz = np.sum(arr1, axis=0)
suma_nieparz = np.sum(arr2, axis =0)
tablica = suma_parz > suma_nieparz
cols = np.array(cols)
array9 = cols[tablica]
