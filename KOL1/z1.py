import numpy as np
import pandas as pd 
data = pd.read_csv(plik.csv) # lub data=pd.read_excel(plik.xlsx)
col = data.columns #Nazwy kolumn do zmiennej
val = data.values #Wartosci do zmiennej 
mean_col = np.mean(val, axis=0) #Srednia dla kazdej kolumny lub chyba mean_col = val.mean(axis=0)
mean_std = np.std(val) #Odychylenie standardowe dla calej tablicy
diffrence = val - mean_std #Od kazdej wartosci tablicy odejmij odychlenie standardowe
max_row_val = np.max(val, axis=1) #Maksymalna wartosc z wiersza
arr2 = val * 2 #Kazdy element
col_np = np.array(col) #konwersja na tablice numpy nazw kolumn
tab_max = col_np[np.max(val) == np.max(val,axis=0)] #Znajdz nazwy kolumn w ktorych znajduje sie wartosc maksymalna
arr9 = (val < mean_std).sum(axis=0) #Sumuj kolumny mniejsze od odchylenia standardowego






##########################################










data=pd.read_excel('practice_lab_1.xlsx')
#ZAD-2.1
cols = data.columns 
values = data.values
arr1= values[::2,:]#Wszystkie kolumny, parzyste wiersze
arr2= values[1::2,:]#Wszystkie kolumny, nieparzyste wiersze
arr = arr1-arr2

#ZAD-2.2
srednia = values.mean()#Srednia dla calej tablicy wartosci
odch = values.std()#Odchylenie standardowe dla calej tablicy wartosci
array3 = (values - srednia ) / odch

#ZAD-2.3
srednia2 = values.mean(axis=0)#Srednia dla kolumn
odch2 = values.std(axis=0)#Odchylenie dla kolumn
array4 = (values - srednia2) / (odch2+ np.spacing(odch2))#np.spacing zeby nie dzielic przez 0

#ZAD-2.4
array5 = (srednia2) / (odch2 + np.spacing(odch2))

#ZAD-2.5
array6 = np.argmax(array5)#Znajdz kolumne (indeks) o najwiekszym wspolczynniku zmiennosci

#ZAD-2.6
array7 = (values > values.mean(axis=0)).sum(axis=0)#Sumuj wartosci wieksze od sredniej kolumn 

#ZAD-2.7
#Znajdz nazwy kolumn w ktorych znajduje sie wartosc maksymalna
max_array = values.max()
max_cols = values.max(axis=0)
cols = np.array(cols)
cols[max_array==max_cols]
#Znajdz nazwy kolumn w ktorych jest najwiecej elementow o wartosci 0
#ZAD-2.8
maska = values == 0
tab_zero = np.sum(maska, axis=0)
max_tabzero = max(tab_zero)
cols = np.array(cols)
print(cols[tab_zero == max_tabzero])
#Znajdz nazwy kolumn w ktorych suma elementow na pozycjach parzystych jest wieksza niz suma na pozycjach nieparzystych
#ZAD-2.9
suma_parz = np.sum(arr1, axis=0)
suma_nieparz = np.sum(arr2, axis =0)
tablica = suma_parz > suma_nieparz
cols = np.array(cols)
array9 = cols[tablica]






#WYKRESY

import numpy as np 
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01) #Zakres i krok

#ZAD 3.1
y = np.tanh(x)
plt.plot(x,y)

# ZAD-3.2 
y = (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
plt.plot(x,y)

# ZAD-3.3
y=(1)/(1+np.exp(-x))
plt.plot(x,y)

# ZAD-3.4
y=np.where(x<=0, 0, x)
plt.plot(x,y)
# ZAD-3.5

y = np.where(x<=0, np.exp(x)-1,x)
plt.plot(x,y)

#ARGMAX ZWRACA INDEKS MAKSYMALNEGO ELEMENTU A MAX WARTOSC!!!