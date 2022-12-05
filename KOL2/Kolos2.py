#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
#ZAD1
data = pd.read_excel('zadanie_1.xlsx')      

def zamien_na_liczbe(data, column, value_to_be_1):
    
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data 

zamien_na_liczbe(data, 'Gender', 'Female')
zamien_na_liczbe(data, 'Married', 'Yes')
zamien_na_liczbe(data, 'Self_Employed', 'Yes')
zamien_na_liczbe(data, 'Education', 'Graduate')
zamien_na_liczbe(data, 'Loan_Status', 'Y')

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis = 1)
data = data.drop(columns = ['Property_Area'])

features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [kNN(n_neighbors=3, weights='distance'), kNN(weights='uniform')]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
#%%

#ZAD2

data = pd.read_csv('zadanie_2.csv', sep=',')
columns = list(data.columns)

def zamien_na_liczbe(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0

zamien_na_liczbe(data, 'label', 'female')

vals = data.values
X = vals[:,:-1]
y = vals[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = np.array(y_train).astype('float')
y_test = np.array(y_test).astype('float')

X_paced=PCA(2).fit_transform(X_train)

females = y_train == 1
fig,ax=plt.subplots(1,1)
ax.scatter(X_paced[females,0], X_paced[females,1], label='female')
ax.scatter(X_paced[~females,0], X_paced[~females,1], label='male')
ax.legend()

pipe = Pipeline([['transformer', PCA(5)],['scaler', RobustScaler()],['classifier', kNN(n_neighbors=3, weights='uniform')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))
#%%