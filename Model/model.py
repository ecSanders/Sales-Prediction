#%% Import libraries
import os
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

PATH = os.getcwd()[:-5]
DATA_PATH = PATH + "Data Analysis\sales EDA.csv"
df = pd.read_csv(DATA_PATH)
df.drop(columns=['Unnamed: 0'], inplace=True)

# %% One Hot Encode
df_cat = df.select_dtypes(include='object')
df.drop(df_cat.columns, axis=1, inplace=True)

ohe = OneHotEncoder()
results = ohe.fit_transform(df_cat).toarray()

df_cat = pd.DataFrame(results, columns=ohe.get_feature_names())
df = pd.concat([df,df_cat],axis=1)
df.dropna(inplace=True)

# %% Split and scale
train_dataset = df.sample(frac=0.80, random_state=1)
train_dataset = (train_dataset[(train_dataset.Order_Year == 2017) & 
        (train_dataset.x9_December == 1)])

X_test = train_dataset.drop(columns=['Profit'])
y_test = train_dataset['Profit']

test_data = df.drop(train_dataset.index)

X_train = test_data.drop(['Profit'], axis=1)
y_train = test_data['Profit']


mms = MinMaxScaler().fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)

#%% Fit, predict, evaluate
xmod = XGBRegressor()
xmod = (xmod.fit(X_train,y_train))
y_hat = xmod.predict(X_test)
r2_score(y_test, y_hat)
mean_squared_error(y_test, y_hat, squared=False)

#%%
xmod.save_model('XGBmodel04.h5')