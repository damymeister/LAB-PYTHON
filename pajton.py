import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#zad1
data_excel=pd.read_excel('practice_lab_1.xlsx')
values = data_excel.values # wartosci
columnes = data_excel.columns #nazwa kolumn 
#ZAD1 #Roznica dwoch tablic
arr1= values[::2,:] #Parzyste wiersze i wszystkie kolumny
arr2= values[1::2,:] #Nieparzyste wiersze i wszystkie kolumny
diff=  arr1-arr2 #Roznica tablic
print(diff)
#ZAD2
srednia= values.mean()# Srednia wartosci tablicy
odchylenie = values.std()#Odchylenie std. wartosci tablicy
array3=(values - srednia) / odchylenie
#ZAD3
srednia2 = values.mean(axis=0)#Srednia dla kolumn
odchylenie2 = values.std(axis=0)#Odchylenie std. dla kolumn
array4=(values - odchylenie2) / (srednia2 + np.spacing(values.std(axis=0)))#Zabezpieczenie przed dzieleniem przez 0
#ZAD4
wspolczynnik = ((srednia2 )/ (odchylenie2 + np.spacing(values.std(axis=0))))
#ZAD5
max= np.argmax(wspolczynnik)#Zwraca indeksy maksymalnych wartosci wzdluz osi
#ZAD6 
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01) #zakres i krok
y = np.tanh(x)
plt.plot(x,y)
import os
#%%