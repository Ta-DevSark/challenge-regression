"""Removing outliers and filling missing values in based on KNN(Nearest neighbours):
pip install scipy,  Scikit-learn, sklearn, run it after outliers are removed"""

import re
import pandas as pd
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

import seaborn as sns

from typing import List

import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#df for houses and apartments: 
df = pd.read_csv("clean_data.csv")

#Deleting manually irrelevant values(like -1 for 'no info)
df["Primary energy consumption"] = np.where((df["Primary energy consumption"] > 10000) | (df["Primary energy consumption"] < 20), 0, df["Primary energy consumption"])

df['Primary energy consumption'] = df['Primary energy consumption'].replace({-1: np.nan,
0 : np.nan, 1:np.nan
})

df['Garden surface'] = df['Garden surface'].replace({-1: np.nan
})

df['Terrace surface'] = df['Terrace surface'].replace({-1: np.nan,
 1:np.nan
})

df['Building Cond. values'] = df['Building Cond. values'].replace({-1: np.nan
})

df['Energy efficiency'] = df['Energy efficiency'].replace({-1: np.nan, 0 : np.nan
})


df['Kitchen values'] = df['Kitchen values'].replace({-1: np.nan
})

#Creating 2 dfs for houses and apartments
df_for_houses = df[df['Type of property'] == 'house']
df_for_apartments = df[df['Type of property'] == 'apartment']

#Modify these columns to include different params 
columns_houses = ['id','Zip','Price','Price of square meter', 'Living area','Number of rooms','Garden surface','Terrace surface','Open fire','Surface of the land','Number of facades','Swimming pool','Building Cond. values','Kitchen values','Primary energy consumption', 'Energy efficiency']
columns_apartments = ['id','Zip','Price','Price of square meter', 'Living area','Number of rooms','Furnished', 'Garden surface', 'Terrace surface', 'Open fire','Building Cond. values','Kitchen values','Primary energy consumption', 'Energy efficiency']

df_houses = df_for_houses[columns_houses]
df_apartments = df_for_apartments[columns_apartments]
df_all = df[columns_houses]

# Remove outliers funtion
def remove_outliers(df: pd.DataFrame, columns: List[str], n_std: int) -> pd.DataFrame:
    for col in columns:
        mean = df[col].mean()
        sd = df[col].std()
        
        df = df[(df[col] <= mean+(n_std*sd))]
        
    return df
# Apply the function: choose the param where you want to remove outliers ('['Price']) and number of standard deviations (3). 
clean_houses = remove_outliers(df_houses, ['Price'], 3)
clean_apartments = remove_outliers(df_apartments, ['Price'], 3)
clean_all = remove_outliers(df, ['Price'], 3)
clean_numeric_all = remove_outliers(df_all, ['Price'], 3)

#Columns for the data frames without 'Price' and 'Price of square meter"

ka_columns = ['id','Living area','Number of rooms','Furnished', 'Garden surface', 'Terrace surface', 'Open fire','Building Cond. values','Kitchen values','Primary energy consumption', 'Energy efficiency']
kh_columns = ['id','Living area','Number of rooms','Garden surface','Terrace surface','Open fire','Surface of the land','Number of facades','Swimming pool','Building Cond. values','Kitchen values','Primary energy consumption', 'Energy efficiency']

k_apartments = clean_apartments[['id', 'Living area','Number of rooms','Furnished', 'Garden surface', 'Terrace surface', 'Open fire','Building Cond. values','Kitchen values','Primary energy consumption', 'Energy efficiency']]

k_houses = clean_houses[['id','Living area','Number of rooms','Garden surface','Terrace surface','Open fire','Surface of the land','Number of facades','Swimming pool','Building Cond. values','Kitchen values','Primary energy consumption', 'Energy efficiency']]


#Filling in the missing data for dfs without price values
from sklearn.impute import KNNImputer
impute_knn = KNNImputer(n_neighbors=5)
k_apartments = impute_knn.fit_transform(k_apartments).astype(int)
k_houses = impute_knn.fit_transform(k_houses).astype(int)

#Creating dfs with missing values filled in 
imputed_houses = pd.DataFrame(k_houses, columns = kh_columns)
imputed_apartments = pd.DataFrame(k_apartments, columns = ka_columns)

#Creating dfs with prices
new_houses = clean_houses[['id', 'Zip', 'Price', 'Price of square meter']]
new_apartments = clean_apartments[['id', 'Zip', 'Price', 'Price of square meter']]

#Merging dfs (with prices and without prices (with other values filled in))
complete_houses = pd.merge(new_houses, imputed_houses,on='id')
complete_apartments = pd.merge(new_apartments, imputed_apartments, on='id')

#complete_apartments.to_csv('complete_apartments.csv')
#complete_houses.to_csv('complete_houses.csv')
