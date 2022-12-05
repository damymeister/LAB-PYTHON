#%% begin
### Import various.shit.*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes
data_excel = pd.read_excel("housing.xlsx")      # Konwersja na DataFrame
korr = data_excel.corr()                        # Macierz korelacji zmiennych
# iloc - metoda pozwalająca wybierać dane z DataFrame za pomocą zwykłego indeksowania Python
# X - Cechy niezlażne, czyli wszystkie dane w kolumach oprócz ostatniej
# metoda shape - zwraca wymiary tablicy, shape[1] zwraca liczbę kolumn
X = data_excel.iloc[:,:data_excel.shape[1]-1]
# Y - Cecha zalezna od pozostałych, czyli ostatnia kolumna (MedianowaCena)
Y = data_excel.iloc[:,-1]
#%% Zadanie 1
# fig - obraz, figsize - szerokość i wysokość obrazu w calach
# ax - siatka wykresów (X.shape[1] {liczba cech niezależnych} wierszy, 1 kolumna wykresów)
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5, 20))
# Generowanie wykresów wyliczając liczbę kolumn (i - licznik od 0, col - nazwa kolumny)
for i, col in enumerate(X.columns):
    # Nanieś wykres punktowy na siatkę ax dla wybranej kolumny (cechy niezależnej)
    ax[i].scatter(X[col], Y)

#%% Zadanie 2
def testuj_model(n):
    s = 0               # Suma błędów modelu
    # Test powtarzamy n razy
    for i in range(n):
        # Losowy podział X, Y na zbiór treningowy oraz testowy
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=221, shuffle=True)
        linReg = LinearRegression()         # Utworzenie nowego obiektu regresji liniowej
        linReg.fit(X_train, y_train)        # Kalkulacja prostej regresji na podstawie zbioru treningowego
        y_pred = linReg.predict(X_test)     # Przewidywanie y na podstawie zbioru testowego X oraz prostej regresji
        s += mean_absolute_percentage_error(y_test, y_pred) # Obliczenie wartości bezwzględnej średniego błędu w % między zbiorem testowym, a przewidywanym
    return s/n                              # Zwracamy średni błąd po wykonaniu n testów

#%% Zadanie 3
def usuniecie (n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=221, shuffle=True)
        ### Obserwacje nietypowe
        # Do poprawienia wyników testów obserwacje odstające można usunąć ze zbioru treningowego
        # Z definicji obserwacje nietypowe uznawane są za te wartości spoza przedziału:
        #   [średnia - 3 * odch_std, średnia + 3 * odch_std]
        # Lub:
        #   wartość < średnia - 3 * odch_std || wartość > średnia + 3 * odch_std
        # Po przekształceniu:
        #   (wartość - średnia) / odch_std < -3 || (wartość - średnia) / odch_std > 3
        # Po skorzystaniu z własności wartości bezwzględnej:
        #   |(wartość - średnia) / odch_std| > 3
        # Wyznaczenie listy obserwacji nietypowych
        outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
        # loc - metoda pozwalająca wybierać dane z DataFrame za pomocą etykiet kolumn (akceptuje listy i obiekty)
        X_train_no_outliers = X_train.loc[~outliers,:]  # X - jako wszystkie cechy niezależne, bez wierszy (obserwacji) nietypowych
        y_train_no_outliers = y_train.loc[~outliers]    # Y - jako cecha zależna bez wierszy (obserwacji) nietypowych
        linReg = LinearRegression()
        linReg.fit(X_train_no_outliers, y_train_no_outliers)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error(y_test, y_pred)
    return s/n

def zamiana (n):
    s = 0
    for i in range(n):
        ### Zamiana - wartości obserwacji odstających zamieniamy wartościami średnimi całego zbioru
        # Zmianie ulegają jedynie cechy ZALEŻNE!
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
        y_train_mean = y_train.copy()           # Kopia zbioru treningowego do nowej zmiennej
        y_train_mean[outliers] = y_train.mean() # Podmiana wartości obserwacji odstających uśrednionymi
        linReg = LinearRegression()
        linReg.fit(X_train, y_train_mean)       # Prostą regresji wpasowywujemy dla zbioru treningowego UŚREDNIONEGO
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error(y_test, y_pred)
    return s/n

#%% Zadanie 5
data = load_diabetes()  # Przykładowe dane z biblioteki sklearn do testowania
### Konwersja na typ DataFrame
# Wartości podane są w obiekcie data z kluczem data
# Nazwy kolumn podane są w obiekcie data z kluczem feature_names
dane = pd.DataFrame(data.data, columns=data.feature_names)
# Wartości cechy zależnej podane są w obiekcie data z kluczem target
# Wartości te dodajemy jako ostatnią kolumnę z indeksem target do tablicy dane
dane['target'] = data.target
# Ciąg dalszy jak w zadaniu 1
kor = dane.corr()
X = dane.iloc[:,:dane.shape[1]-1]
Y = dane.iloc[:,-1]
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], Y)
testuj_model(10)
usuniecie(10)
zamiana(10)
# %%











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
data = load_diabetes()
plik = pd.DataFrame(data.data, columns = data.feature_names)
plik['target'] = data.target #Wartosc cechy zaleznej podana jest w obiekce data z kluczem target
koleracja = plik.corr() #Tworzymy macierz korelacji
X = plik.iloc[:,:plik.shape[1]-1]#Shape zwraca wymiary tab, shape[1] liczbe kolumn
Y = plik.iloc[:,-1]#Iloc sprawia ze mozna zastosowac zwykle indeksowanie
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20)) #fig-obraz, ax-siatka obrazow, x.shape[1] liczba cech niezaleznych, 1 kolumna wykresow
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], Y)#Dla wybranej cechy niezaleznej(kolumny) wykres punktowy

def test(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=221, shuffle=True)#Losowy podzial na zbior treningowy i testowy        
        linearReg = LinearRegression()
        linearReg.fit(X_train, y_train)#Kalukacja regresji (zbior treningowy)
        y_predict = linearReg.predict(X_test)#Przewidywanie y na podstawie x_testowego
        s += mean_absolute_percentage_error(y_test, y_predict)#Obliczanie wartosci bezwzg. w % miedzy zbiorem testowym a przewidywanym
    return s/n
def usun(n):
    s=0
    for i in range(n):
         X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=221, shuffle=True)
         outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
         X_train_no_outliers = X_train.loc[~outliers,:]#Wszystkie cechy niezalezne bez nietypowych
         Y_train_no_outliers = y_train.loc[~outliers]#Cecha zalezna bez nietypowych
         linearReg = LinearRegression()
         linearReg.fit(X_train_no_outliers, Y_train_no_outliers)
         y_predict = linearReg.predict(X_test)
         s+=mean_absolute_percentage_error(y_test, y_predict)
    return s/n
#Wartosci obserwacji odstajacych zastepujemy srednimi calego zbioru, zmieniamy tylko cechy zalezne
def zastap(n):
    s=0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=221, shuffle=True)
        outliers = np.abs((y_train - y_train.mean())/ y_train.std()) > 3
        y_train_mean = y_train.copy()#Kopiujemy zbior treningowy
        y_train_mean[outliers] = y_train.mean()#Zamiana odstajacych wartosci na usrednione wartosci
        linearReg= LinearRegression()
        linearReg.fit(X_train, y_train_mean )#Liczymy ze zbiorem treningowym usrednionym
        y_predict = linearReg.predict(X_test)
        s+=mean_absolute_percentage_error(y_test, y_predict)
    return s/n
test(9)
usun(9)
zastap(9)