#%% Import libraries
import os
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

PATH = os.getcwd()[:-5]
DATA_PATH = PATH + "Data Analysis\sales EDA.csv"

df = pd.read_csv(DATA_PATH)
drop_lst = (['Row ID', 'Unnamed: 0',
            'Order Date', 'Ship Date', 
            'Order ID', 'Product Name',
            'Customer Name','Country'])
df.drop(drop_lst, axis=1, inplace=True)

#%%
df = df[(df.mon)& ()]

#%%
y = df.Profit
X = df.drop(['Profit'], axis=1)

# %%
X_cat = X.select_dtypes(include='object')
X.drop(X_cat.columns, axis=1, inplace=True)

X_cat = OneHotEncoder().fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)
X = pd.concat([X,X_cat],axis=1)

# %%
