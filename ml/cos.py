import csv
import pandas as pd
import numpy as np
# df_2017 = pd.read_csv('C:\Ml_Assignments\ml\dane2017.csv') 
df_2018 = pd.read_csv('dane2017.csv') 
columns_to_replace = ['MOONPHASE', 'CONDITIONS', 'CLOUDCOVER', 'VISIBILITY', 'PRESSURE', 
                       'WINDDIR', 'WINDSPEED', 'WINDGUST', 'PRECIPTYPE', 'SNOWDEPTH', 
                       'SNOW', 'PRECIPPROB', 'PRECIP', 'DEW', 'HUMIDITY', 'TEMP', 'FEELSLIKE']

def Delaycollcreate():
    df['DELAY'] = ((df['ARR_TIME'] - df['CRS_ARR_TIME'] > 0) | (df['ARR_TIME'] < df['CRS_DEP_TIME'])).astype(int)

for i in range(359):
    subset_df = df_2018[df_2018["ORIGIN"] == i]
    
    filename = f'C:\Ml_Assignments\ml\originsplited_2017\df_2017_{i}.csv'
    
    subset_df.to_csv(filename,index_label='ID')
