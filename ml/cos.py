import csv
import pandas as pd
import numpy as np
df_2017 = pd.read_csv("dane2017.csv")
df_2018 = pd.read_csv("dane2018.csv")
def Delaycollcreate():
    df['DELAY'] = ((df['ARR_TIME'] - df['CRS_ARR_TIME'] > 0) | (df['ARR_TIME'] < df['CRS_DEP_TIME'])).astype(int)

columns_to_replace = ['MOONPHASE', 'CONDITIONS', 'CLOUDCOVER', 'VISIBILITY', 'PRESSURE', 
                       'WINDDIR', 'WINDSPEED', 'WINDGUST', 'PRECIPTYPE', 'SNOWDEPTH', 
                       'SNOW', 'PRECIPPROB', 'PRECIP', 'DEW', 'HUMIDITY', 'TEMP', 'FEELSLIKE']

print(df_2017.info())