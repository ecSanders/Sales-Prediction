#%% Import libraries
import os
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler

PATH = os.getcwd()[:-5]
DATA_PATH = PATH + "Data Analysis\sales EDA.csv"

df = pd.read_csv(DATA_PATH)
drop_lst = (['Row ID', 'Unnamed: 0',
            'Order Date', 'Ship Date', 
            'Order ID', 'Product Name',
            'Customer Name','Country',
            'Ship_DOW','Ship_Year',
            'Ship_Month', 'Customer ID',
            'Product ID'])
df.drop(drop_lst, axis=1, inplace=True)


# %% One Hot Encode
df_cat = df.select_dtypes(include='object')
df.drop(df_cat.columns, axis=1, inplace=True)

ohe = OneHotEncoder()
results = ohe.fit_transform(df_cat).toarray()

df_cat = pd.DataFrame(results, columns=ohe.get_feature_names())
df = pd.concat([df,df_cat],axis=1)
df.dropna(inplace=True)

# %%
train_dataset = df.sample(frac=0.85, random_state=42)
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

#%%
xmod = XGBRegressor()
xmod = xmod.fit(X_train,y_train)
y_hat = xmod.predict(X_test)
r2_score(y_test, y_hat)
#%%
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')

  # Examine more closely
  

  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

# Learnrate schedule
def sketch(epoch, lr):
  if epoch >= 500:
    return 0.001
  if epoch >= 440:
    return 0.002
  if epoch >= 100:
    return 0.003
  else:
    return lr

# Build model
def build_model(norm):
    model = keras.Sequential([
        layers.Dense(2048, activation=tf.nn.leaky_relu),
        layers.Dense(2048, activation=tf.nn.leaky_relu),
        layers.Dense(1024, activation=tf.nn.leaky_relu),
        layers.Dense(1)
    ])    
        
    (model.compile(optimizer=tf.optimizers.Adam(0.0001) , 
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError()]))
    return model

# Sub b/c I'm lazy
normalizer = 0
model = build_model(normalizer)

# Fit and stuff
callback = keras.callbacks.LearningRateScheduler(schedule=sketch)
history = model.fit(
    X_train,
    y_train,
    # batch_size=250,
    validation_split=0.2,
    epochs = 100,
    # callbacks=[callback],
    verbose=2
)

plot_loss(history)
# %%
y_hat = model.predict(X_test)
r2_score(y_test,y_hat)
# %%
