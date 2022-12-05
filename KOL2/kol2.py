#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


data = pd.read_excel('loan_data.xlsx')     
columns = list(data.columns)    

def tekst_na_binarke(data, column, value_to_be_1):
    
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data 

tekst_na_binarke(data, 'Gender', 'Female')
tekst_na_binarke(data, 'Married', 'Yes')
tekst_na_binarke(data, 'Self_Employed', 'Yes')
tekst_na_binarke(data, 'Education', 'Graduate')
tekst_na_binarke(data, 'Loan_Status', 'Y')


cat_feature = pd.Categorical(data.Property_Area)#W to miejsce wpisujemy kolumne ktora ma wiecej cech "slownych" niz 1
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])#Tutaj wpisujemy kolumne ktora wpisalismy wyzej


features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)##Wpisujemy kolumne wyjsciowa
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)#Wpisujemy kolumne wyjsciowa

#Podzial zbioru na wejscie wyjcie, u nas wyjsciem jest kolumna Loan_status
features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].values#Tu jest kolumna wyjsciowa
X = data.drop(columns = ['Loan_Status']).values#Tutaj tez jet kolumna wyjsciowa

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

y_train = np.array(y_train).astype('float') #Nie wiadono czy trzeba ale ponoc tylko z tym dziala(raczej tak) 
y_test = np.array(y_test).astype('float') #Nie wiadomo czy trzeba ale ponoc tylko z tym (raczej tak)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [SVM(kernel='sigmoid'), kNN(weights='distance',)]#dla SVM kernel:{‘linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} - domyslnie rbf
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
#%%
#ZAD2

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
data = pd.read_csv('voice_extracted_features.csv', sep=',')
columns = list(data.columns)

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0

qualitative_to_0_1(data, 'label', 'female')

vals = data.values
X = vals[:,:-1]
y = vals[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()#PODOBNO TRZEBA
X_train = scaler.fit_transform(X_train)#PODOBNO TRZEBA

y_train = np.array(y_train).astype('float')#NIE WIEM  CZY MUSI BYC(raczej nie)
y_test = np.array(y_test).astype('float')#NIE WIEM CZY MUSI BYC(raczej nie)

X_paced=PCA(2).fit_transform(X_train)

females = y_train == 1
fig,ax=plt.subplots(1,1)
ax.scatter(X_paced[females,0], X_paced[females,1], label='female')
ax.scatter(X_paced[~females,0], X_paced[~females,1], label='male')
ax.legend()

from sklearn.pipeline import Pipeline
pipe = Pipeline([['transformer', PCA(7)],['scaler', MinMaxScaler()],['classifier', kNN(weights='distance')]])#dla knn weights moze byc uniform albo distance, n_neighbors domyslnie 5
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

confusion_matrix(y_test, y_pred)

#%%