from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler    # Standardowy Skaler, Min-Max Skaler, Solidny Skaler
from sklearn.metrics import confusion_matrix                                    # Matrix Zdziwienia
from sklearn.neighbors import KNeighborsClassifier as kNN                       # K-Sąsiad klasyfikator             
from sklearn.model_selection import train_test_split                            # Ćwicz, testuj, podziel
from sklearn.svm import SVC as SVM                                              # Maszyny wektorów nośnych
import pandas as pd                                                             # Pandy
import numpy as np                                                              # nampaj
# %% Zadanie 3.2
data = pd.read_excel('loan_data.xlsx')                  # Nudy
columns = list(data.columns)                            # Nudy
def qualitative_to_0_1(data, column, value_to_be_1):    # Funkcja która zamienia cechy na wartości binarne
    mask = data[column].values == value_to_be_1     # Maska na dane
    data[column][mask] = 1                          # Jak w masce true to 1 wstawiamy
    data[column][~mask] = 0                         # Jak false to 0
    # Nie ma return data bo kurwa data jest zmienną GLOBALNĄ
qualitative_to_0_1(data, 'Gender', 'Female')            # Kolumna Gender - Jak wystąpi Female to zastąpi to 1 jak Male 0
qualitative_to_0_1(data, 'Married', 'Yes')              # Married Yes - 1, Kasztan - 0
qualitative_to_0_1(data, 'Self_Employed', 'Yes')        # itd... zmieniamy
qualitative_to_0_1(data, 'Education', 'Graduate')
qualitative_to_0_1(data, 'Loan_Status', 'Y')
# %% Zadanie 3.3
def Metrics (tp, fp, tn, fn):                       # Funkcja do liczenia i wyświetlania metryk
    sensivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    specifity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    # Żeby debil przez 0 nie dzielił
    if sensivity + precision != 0: F1 = (2 * sensivity * precision) / (sensivity + precision)
    else: F1 = 0
    print("Sensivity: " + str(sensivity))
    print("Precision: " + str(precision))
    print("Specifity: " + str(specifity))
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(F1))
Metrics(7, 26, 17, 73)
Metrics(0, 33, 0, 90)
# %% Zadanie 3.4
# Cech jakościowych nie da się zamienić na wartości 0/1 ponieważ mogą mieć one wiele wartości
# Wykorzystuje się do tego podejście Jeden-Gorący (kodowanie 1-z-n)
cat_feature = pd.Categorical(data.Property_Area)            # Konwersja cechy jakościowej na zmienną kategorialną
one_hot = pd.get_dummies(cat_feature)                       # Wybranie Jednego-Gorącego
data = pd.concat([data, one_hot], axis=1)                   # Dołączamy nowo utworzoną kolumnę do danych
data = data.drop(columns=['Property_Area'])                 # Wywalamy cechę jakościową i git
vals = data.values.astype(np.float64)                       # Konwersja na floaty
y = data['Loan_Status'].values                              # y - wartości w Loan_Status
X = data.drop(columns = ['Loan_Status']).values             # X - Cała reszta (wszystkie wartości wywalając Loan_Status)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=True)    # To z Lab2
y_train = np.array(y_train).astype('int')                   # Jakaś tam konwersja żeby działało
y_test = np.array(y_test).astype('int')                     
models = [SVM(), kNN(n_neighbors=4, weights="distance")]    # 2 modele - w kNN można zmieniać sobie parametry
for model in models:                                        # Pętla dla modeli
    model.fit(np.array(X_train), y_train)                   # Wpasowanie modelu
    y_pred = model.predict(X_test)                          # Przeiwdziane wartości
    print(confusion_matrix(y_test, y_pred))                 # Macierz pomyłek
# %% Zadanie 3.5
scaler = StandardScaler()                                   # Nowy skaler
scaler.fit(X_train)                                         # Dopasowanie skalera do danych
X_train = scaler.transform(X_train)                         # Transformacja zbioru treningowego
X_test = scaler.transform(X_test)                           # i testowego
models = [SVM(), kNN(n_neighbors=4, weights="distance")]    # Znowu trzeba modele wpasować
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
# itd...
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [SVM(), kNN(n_neighbors=4, weights="distance")]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    
scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
models = [SVM(), kNN(n_neighbors=4, weights="distance")]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

# %% Zadanie 3.6
# Kolejny Import
from sklearn.tree import DecisionTreeClassifier as DT       # Klasyfikator Drzewa Decyzyjnego
from sklearn.tree import plot_tree                          # Wykres Drzewo
from sklearn.datasets import load_breast_cancer             # Załaduj raka piersi
data = load_breast_cancer()                                 # Pobierz randomowe dane
dane = pd.DataFrame(data.data, columns=data.feature_names)  # Z Lab2 - zamień na DataFrame, nazwy kolumn w feature_names
dane['target'] = data.target                                # Dodaj wartości docelowe do danych
X = dane.iloc[:,:dane.shape[1]-1]                           # X - wszystkie cechy
y = dane.iloc[:,-1]                                         # y - target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=True) # tak tak
model = DT(max_depth=5)                                     # Model drzewa o głębokości 5
model.fit(X_train, y_train)                                 # Wpasuj model
y_pred = model.predict(X_test)                              # Przewidzenie wartości
cm = confusion_matrix(y_test, y_pred)                       # Można se zobaczyć macierz konfuzji
print(cm)
from matplotlib import pyplot as plt                        # PyWykres
plt.figure(figsize=(40,30))                                 # 40x30 cali obrazek
# Drzewo modelu z nazwami cech jak w danych, nazwami klas (cokolwiek to jest) N, Y, czcionka 20
tree_vis = plot_tree(model, feature_names=data.feature_names[:-1], class_names=['N', 'Y'], fontsize = 20)
# %%