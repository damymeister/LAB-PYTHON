import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
#zad1
data=pd.read_excel('practice_lab_1.xlsx')
cols = data.columns
vals = data.values
arr1 = vals[::2,:]
arr2 = vals[1::2,:]
diff= arr1 - arr2


#1.2

avg = vals.mean()
sr= vals.std()
array3=(vals-avg)/sr

#1.3

diff=vals.std(axis=0)
avg2 = vals.mean(axis=0)
array4 = (vals- avg2)/(diff+ np.spacing(arr.std(axis=0)))

#1.4

array5= ((diff)/(avg+np.spacing(vals.std(axis=0))))

#1.5
zad5 = np.argmax(array5)
#1.6
zad6=(vals>vals.mean(axis=0)).sum(axis=0)
#1.7
max_value= vals.max()
col_max = vals.max(axis=0)
cols = np.array(cols)#rzutowanie tablicy 
cols[col_max ==max_value] #wyswietlanie 
#1.8 i #1.9 do domu

#1.3 nowe zad wykresy
x = np.arrange(-5,5,0.01) #zakres i krok
y = np.tanh(x)
plt.plot(x,y)

#ex-1, x<=0
plt.plot(x[x>0, x[x>0]])
#plt.plot(x[x<=0], np.exp(x[]
#pd 2 3 i 4 z 1.3
# i 1.4