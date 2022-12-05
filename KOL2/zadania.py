#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC

data = pd.read_excel('loan_data.xlsx')

def qualitative_to_0_1(data, column, value_to_be_1):
    
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data 

data = qualitative_to_0_1(data, 'Gender', 'Female')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

#zamiana cech jakosciowych na zmienna kategorialna
cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])
#koniec zamiany

#podzial zbioru na wejscie/wyjscie
features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)
#koniec podzialu na wejscie wyjscie

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
#zadanie 3.4
from sklearn.svm import SVC as SVM
models = [kNN(weights='distance'), SVM(kernel = 'sigmoid')] 
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
#Zadanie 3.5
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#to musimy znowu zaladowac
features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#dotad

scaler_minMaxScaler = MinMaxScaler()
scaler_minMaxScaler.fit(X_train)
X_train = scaler_minMaxScaler.transform(X_train)
X_test = scaler_minMaxScaler.transform(X_test)

from sklearn.svm import SVC
models = [kNN(), SVC()]
for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
#%%
#ZAD4.1

data = pd.read_csv("voice_extracted_features.csv", sep=',')
data = qualitative_to_0_1(data, 'label', 'female')
features = list(data.columns)
val = data.values

X = val[:,:-1]
Y = val[:,-1]

#1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#2

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
X_paced=PCA(2).fit_transform(X_train)
fig,ax=plt.subplots(1,1)
females=y_train==1
ax.scatter(X_paced[females,0],X_paced[females,1],label='female')
ax.scatter(X_paced[~females,0],X_paced[~females,1],label='male')
ax.legend()

#3

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))
PC_num = (cumulated_variances<0.95).sum()

#4
from sklearn.pipeline import Pipeline
pipe =Pipeline([['transformer', PCA(9)],['scaler', StandardScaler()],['classifier', kNN(weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

#%%
