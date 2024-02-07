# -*- coding: utf-8 -*-
import tqdm
import os
import sys
import opendatasets as od
from joblib import parallel_backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sopt
import scipy.stats as sstats
import seaborn as sns
import sklearn.ensemble
import sklearn.tree
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from matplotlib import animation, pyplot, rc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.utils import resample
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from datetime import datetime, time
import json 
#import 
df = pd.read_csv("dane2018.csv")
pd.set_option("display.max_columns", None)
def transformationintdata(intdata):
    godzina = intdata // 100  
    minuty = intdata % 100  
    czas_obiekt = time(godzina, minuty)
    return czas_obiekt
def normalizacja(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
def equaldelay(df,name):
    df_class_0 = df[df['Delay'] == 0]
    df_class_1 = df[df['Delay'] == 1]
    min_samples = min(len(df_class_0), len(df_class_1))
    delay_0_selected = df_class_0.sample(n=min_samples, random_state=42)
    delay_1_selected = df_class_1.sample(n=min_samples, random_state=42)

# Połącz wybrane dane dla obu grup
    balanced_data = pd.concat([delay_0_selected, delay_1_selected])
    balanced_data.to_csv(name)
    
    
# IN JUPYTER ##
def check_nans(df):
  # since all len's are same
  n = len(df['FL_YEAR'])
  for col in df.columns.values:
    nans = df[col].isna().sum()
    print(f"Col name: {col:<21} nans: {nans:<10}   {round((nans / n) * 100,2)} %")
  print(f"\nThere are {n} rows")
  
  
def mapping(df,col_name,filename):

    
    with open('mappings.json', 'r') as file:
        mapowanie_preciptype = json.load(file)
    dic_kind = mapowanie_preciptype[col_name]
    my_mapping = {key if key != 'nan' else np.nan: value for key, value in dic_kind.items()}

    df[col_name] = df[col_name].map(my_mapping)

    df.to_csv(filename, index=False)
    # df['preciptype'] = df['preciptype'].astype(str)
    # df['conditions'] = df['conditions'].astype(str)
    # df['ORIGIN'] = df['ORIGIN'].astype(str)

    # m1 = make_mapping('preciptype')
    # m2 = make_mapping('conditions')
    # m3 = make_mapping('ORIGIN')

    
#wykresy
def wykreskolowydelay(df):
    delayed_flights = df[df['Delay'] > 0]
    percentages = [len(delayed_flights) / len(df) * 100, 100 - len(delayed_flights) / len(df) * 100]
    labels = ['Opóźnione', 'Bez opóźnień']
    plt.figure(figsize=(8, 8))
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], startangle=90)
    plt.title('Procent opóźnionych lotów')
    plt.show()
def wykresmiesiacdealay(df):
    monthly_delays = df.groupby(df['FL_MONTH'])['Delay'].mean() * 100
    plt.figure(figsize=(10, 6))
    monthly_delays.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Procentowe opóźnienia')
    plt.xlabel('Miesiąc')
    plt.ylabel('Procentowe opóźnienie')
    plt.xticks(rotation=0)
    plt.show()
def wykresmiesiacliczbalotow(df):
    monthly_delays = df.groupby(df['FL_MONTH'])['id'].count()
    print(monthly_delays)
    plt.figure(figsize=(10, 6))
    monthly_delays.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Liczba lotów na miesiąc')
    plt.xlabel('Miesiąc')
    plt.ylabel('Liczba lotów')
    plt.xticks(rotation=0)
    plt.show()
def wykresmiesiacopoznienie(df):
    monthly_delays = df.groupby('FL_MONTH')['Delay'].sum()
    plt.figure(figsize=(10, 6))
    print(monthly_delays)
    monthly_delays.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Ilość opóźnionych lotów')
    plt.xlabel('Miesiąc')
    plt.ylabel('Liczba opóźnień')
    plt.xticks(rotation=0)
    plt.show()

def wykresPCA(a):
    model=PCA(a)
    model.fit(X_train,y_train)
    explained_var_ratio = model.explained_variance_ratio_

    
    cumulative_var_ratio = np.cumsum(explained_var_ratio) * 100

    print(f"PCA Components: {a}")
    for i, var_ratio in tqdm(enumerate(explained_var_ratio)):
        print(f"Komponent {i + 1}: {var_ratio * 100:.2f}% wariancji")
    print(sum(explained_var_ratio) * 100)
def wykreskorelacja_pogoda(df):
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(pogoda, cmap='coolwarm', square=True)
    # plt.show()
    korelacje = df.corrwith(df['ARR_DELAY'])
    print(korelacje.sort_values(ascending=False))
# target = df['Delay']

# features = df.drop(columns=['ARR_DELAY', 'ARR_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DEP_DELAY', 'DEP_DELAY','Delay'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
# normalizacja(X_train,X_test)

#modele
def LogREG(X_train,X_test,y_train,y_test):
    model = LogisticRegression(max_iter=1000)

    # Different logistic regression parameters to try
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag']}

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model from the grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("LR best parameters:", best_params)
    print("Accuracy:", accuracy)
def Principalcomponentanalysis(X_train,X_test,y_train,y_test):
    pcasplitnum = range(2, X_train.shape[1], 1)  # Zmieniłem 'features' na 'X_train.shape[1]'
    knnpar = 5
    best_accuracy = 0.0
    best_pcasplit = 0

    for pcasplit in pcasplitnum:
        pipeline = Pipeline([
            ('pca', PCA(n_components=pcasplit)),
            ('knn', KNeighborsClassifier(n_neighbors=knnpar))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        #print(f"PCA Components: {pcasplit}, KNN Neighbors: {knn_param}, Accuracy: {accuracy}")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_pcasplit = pcasplit

    print("\nNajlepsza para parametrów:")
    print(f"PCA Components: {best_pcasplit}, KNN Neighbors: {knnpar}")
    print(f"Najlepszy wynik accuracy: {best_accuracy}")
def SVM(X_train, X_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Dokładność klasyfikacji: {accuracy:.2f}')

    # Wyświetlanie pełnego raportu klasyfikacji
    print('\nRaport klasyfikacji:')
    print(classification_report(y_test, y_pred))
def DecisionTree(X_train, X_test, y_train, y_test, depth=4):

    # Inicjalizacja klasyfikatora drzewa decyzyjnego
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)

    # Trenowanie klasyfikatora na danych treningowych
    clf.fit(X_train, y_train)

    # Przewidywanie etykiet dla danych testowych
    y_pred = clf.predict(X_test)

    # Ocena dokładności klasyfikatora
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Dokładność klasyfikacji: {accuracy:.2f}')

    # Wyświetlanie pełnego raportu klasyfikacji
    print('\nRaport klasyfikacji:')
    print(classification_report(y_test, y_pred))          
def GDA(X_train, X_test, y_train, y_test):

    # Inicjalizacja klasyfikatora GDA
    clf = GaussianNB()

    # Trenowanie klasyfikatora na danych treningowych
    clf.fit(X_train, y_train)

    # Przewidywanie etykiet dla danych testowych
    y_pred = clf.predict(X_test)

    # Ocena dokładności klasyfikatora
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Dokładność klasyfikacji: {accuracy:.2f}')

    # Wyświetlanie pełnego raportu klasyfikacji
    print('\nRaport klasyfikacji:')
    print(classification_report(y_test, y_pred))

