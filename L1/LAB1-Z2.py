#%%
import numpy as np 
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.01) #Zakres i krok

#%% ZAD 3.1
y = np.tanh(x)
plt.plot(x,y)

#%% ZAD-3.2 
y = (np.exp(x)-np.exp(-x))/(np.exp(x) + np.exp(-x))
plt.plot(x,y)

#%% ZAD-3.3
y=(1)/(1+np.exp(-x))
plt.plot(x,y)

#%% ZAD-3.4
y=np.where(x<=0, 0, x)
plt.plot(x,y)

#%% ZAD-3.5
y = np.where(x<=0, np.exp(x)-1,x)
plt.plot(x,y)
# %%
