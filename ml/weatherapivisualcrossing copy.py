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
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from datetime import datetime, timedelta
import requests
import ast
import json
import time
import csv
import threading

gc.enable()
def timeconvert(year,month,day,time_as_int):
  
    time_as_str = str(time_as_int).zfill(4)
    formatted_time = f"{time_as_str[:2]}:{time_as_str[2:]}"
    date_to_convert = datetime(year,month,day)
    start = f"{date_to_convert.strftime('%Y-%m-%d')}T{formatted_time}:00"
    return start
def pobierz_wiersz_z_csv( numer_wiersza):
    return df.iloc[numer_wiersza].tolist()
def zapisz_modyfikacje_do_csv( numer_wiersza, nowy_wiersz):
    df.loc[numer_wiersza] = nowy_wiersz

   
def mapping(val,whattomap):
    with open('mappings.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        dict = data.get(whattomap)
        respdict = dict.get(str(val)) 
    return respdict
def pobranie_danych(grid,rok):
    print("pobieranie")
    with open('pass.txt', 'r') as plik:
        API_key = plik.read().strip()
        
    for orig in range(2,numoforigins):#zaczynam od 2 bo 0 1 pobralem juz 
        
        originname=mapping(orig,"ORIGIN") #do zrobienia mapping ale nie mam slownika 
        cityname = originname["City"]
        state= originname["State"]
        URL='https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'+cityname+','+state+'/'+str(rok)+'-01-01/'+str(rok)+'-12-31?key='+API_key 
        response=requests.get(URL)
        print( response.raise_for_status())
        weather_day_data = response.json()
        for day in range(numofdaysinyear):
            squer=grid[orig][day]
            print(squer,orig,day)
            if len(squer)>1:
                #print("chuj")
                for hours in squer[1:]:
                    indeks_wiersza = hours[0]
                    hour=hours[1]
                    hour=hour//100
                    
                    desired_hour_data = weather_day_data["days"][day]["hours"][hour] #extracting the hour data
                   # print()
                    important_data = {
                            13: weather_day_data["days"][day].get('moonphase', None),
                            14: desired_hour_data.get('conditions', None),
                            15: desired_hour_data.get('cloudcover', None),
                            16: desired_hour_data.get('visibility', None),
                            17: desired_hour_data.get('pressure', None),
                            18: desired_hour_data.get('winddir', None),
                            19: desired_hour_data.get('windspeed', None),
                            20: desired_hour_data.get('windgust', None),
                            #21: desired_hour_data.get('preciptype', None),#tu jest problem ze zwraca tablice wtedy wpierdala a tez latwo nie da sie sformatowac bo czasami zwraca nana mapowac tez nie mozna bo jest wiecej kombinacji niz mamy zapisne
                            22: desired_hour_data.get('snowdepth', None),
                            23: desired_hour_data.get('snow', None),
                            24: desired_hour_data.get('precipprob', None),
                            25: desired_hour_data.get('precip', None),
                            26: desired_hour_data.get('dew', None),
                            27: desired_hour_data.get('humidity', None),
                            28: desired_hour_data.get('temp', None),
                            29: desired_hour_data.get('feelslike', None)
                                    }
                    line=pobierz_wiersz_z_csv(indeks_wiersza) 
                    #print(line)
                    for key, value in important_data.items():
                        line[key]= value
                    #print(line)
                    zapisz_modyfikacje_do_csv(indeks_wiersza,line)
        print(orig)
        df.to_csv(dataname,index=False)

def daycode(year,month,day,baseyear):
    data_formatowana = datetime(year, month, day)
    data_bazowa = datetime(baseyear, 1, 1)
    roznica = data_formatowana - data_bazowa
    return roznica.days
def gridmaker(year):
    print("in")
    grid = [[[] for _ in range(numofdaysinyear)] for _ in range(numoforigins)]
    for index, row in df.iterrows():
        #print(row.iloc[8], row.iloc[9], row.iloc[10])
        if int(row.iloc[8])!=year:
            continue
        nrofday = daycode(int(row.iloc[8]), int(row.iloc[9]), int(row.iloc[10]),year)
        #print(int(row.iloc[8]),nrofday)
        orig = int(row.iloc[1])
        deptime = int(row.iloc[3])
        #im adding year, month, day on the begining of the list and next date values are with row index 
        if grid[orig][nrofday] == []:
            
            grid[orig][nrofday].append((int(row.iloc[8]), int(row.iloc[9]), int(row.iloc[10])))
            #print(grid[orig][nrofday],orig,nrofday)
        grid[orig][nrofday].append((index,deptime))
        #print(grid[orig][nrofday],orig,nrofday)
        #print(grid[carrier][nrofday])
    print("out")
    return grid

numoforigins=359
numofdaysinyear=365
dataname='dane2018.csv'
df = pd.read_csv(dataname) # nany jako -1 i wszsytko int zeby nie bylo bladow w daycode
pobranie_danych(gridmaker(2018),2018)
df.to_csv(dataname,index=False)


