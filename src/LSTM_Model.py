#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:27:11 2019

@author: seth
"""
import quandl
import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import backend as K

plt.style.use('fivethirtyeight')
quandl.ApiConfig.api_key = os.environ.get('quandl')

df = quandl.get('WIKI/MSFT',rows=1305)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def _scale_and_shape(df):
    data = np.array(df['Close'].values).reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X_train = [] 
    y_train = []
    for i in range(261,len(data)-261):
        X_train.append(scaled_data[i-261:i,0])
        y_train.append(scaled_data[i,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    return(X_train,y_train,scaler)
    
def make_LSTM(df):
    X_train,y_train,scaler = _scale_and_shape(df)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(loss=root_mean_squared_error, optimizer='adam')
    model.fit(X_train, y_train, epochs=1, batch_size=32)

    return(model,scaler)
    
def predict(model,scaler):
    inputs = df['Close'][783:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(261,inputs.shape[0]):
        X_test.append(inputs[i-261:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    return closing_price

def graph(model,df,scaler):
    predicts = predict(model,scaler)
    train = pd.DataFrame()
    train['Close'] = df['Close'][:1044]
    valid = pd.DataFrame()
    valid['Close'] = df['Close'][1044:1305:]
    valid['Predict'] = predicts

    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predict']])
    plt.legend(['History','Actual','Predict'])
    plt.title('LSTM Predictions 1 Year')
model,scaler = make_LSTM(df)
predicts = predict(model,scaler)
graph(model,df,scaler)

a = df['Close'][1044:1305:].values
b = predicts
rsme = np.sqrt(np.mean((a-b)**2))