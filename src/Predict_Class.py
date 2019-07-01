#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:18:34 2019

@author: seth
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
from fbprophet import Prophet
from suppress import suppress_stdout_stderr
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import backend as K

plt.style.use('fivethirtyeight')
quandl.ApiConfig.api_key = os.environ.get('quandl')

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
class Prophet_Model():
    
    def __init__(self,exchange,symbol,days_to_grab=-1):
        self.exchange = exchange.upper()
        self.symbol = symbol.upper()
        self.rows = days_to_grab
        self.get = exchange + '/' + symbol
        self._get_stock()
        
    def _get_stock (self):
        if self.rows == -1:
            self.stock = quandl.get(self.get)
        else:
            self.stock = quandl.get(self.get,rows=self.rows)


class LSTM_Model():
    
    def __init__(self,exchange,symbol,days_to_grab=-1):
        self.exchange = exchange.upper()
        self.symbol = symbol.upper()
        self.rows = days_to_grab
        self.get = exchange + '/' + symbol
        self._get_stock()
        self._scale_and_fit()
        
    def _get_stock (self):
        if self.rows == -1:
            self.stock = quandl.get(self.get)

        else:
            self.stock = quandl.get(self.get,rows=self.rows)
    def _scale_and_fit (self):
        data = np.array(self.stock['Close'].values).reshape(-1,1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        X_train = [] 
        y_train = []
        for i in range(261,len(data)-261):
            X_train.append(scaled_data[i-261:i,0])
            y_train.append(scaled_data[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
    
        model.compile(loss=root_mean_squared_error, optimizer='adam')
        model.fit(X_train, y_train, epochs=1, batch_size=32)
