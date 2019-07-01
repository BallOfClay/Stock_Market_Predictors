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
    
    def __init__(self,exchange,symbol,days_to_grab=1305,days_to_predict=261):
        self.exchange = exchange.upper()
        self.symbol = symbol.upper()
        self.rows = days_to_grab
        self.get = exchange + '/' + symbol
        self.days_predict = days_to_predict
        self._get_stock()
    
    def plot_hist(self):
        plt.plot(self.stock['Close'])
        plt.title(self.symbol+' Stock History')
        plt.xlabel('Date')
        plt.ylabel('Value (US$)')       

    def _get_stock (self):
        if self.rows == -1:
            self.stock = quandl.get(self.get)
        else:
            self.stock = quandl.get(self.get,rows=self.rows)
        self._fit()
        
    def _fit (self):
        X = self.stock.index
        y = self.stock.Close
        train = pd.DataFrame()
        train['y'] = y.values
        train['ds'] = X.values
        with suppress_stdout_stderr():
            model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=False,
                            yearly_seasonality=True,
                            changepoint_prior_scale=.05
                            )
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(train)
            future = model.make_future_dataframe(periods=self.days_predict)
            self.forecast = model.predict(future)
            
    def show(self):
        graph = pd.DataFrame(index=self.stock.index[-self.days_predict:])
        graph['History'] = self.stock[-self.days_predict:].Close
        predicted = self.forecast[-self.days_predict:]
        predicted.set_index('ds',inplace=True)
        plt.plot(graph)
        plt.plot(predicted['yhat'])
        plt.legend(['History','Predicted'])
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Value (US$0)')
        plt.title(self.symbol + ' prediction')
        '''
        x = self.stock[-self.days_predict:].Close
        y = self.forecast[-self.days_predict:].yhat
        '''
        
class LSTM_Model():
    
    def __init__(self,exchange,symbol,days_to_grab=1305,days_to_predict=261):
        self.exchange = exchange.upper()
        self.symbol = symbol.upper()
        self.rows = days_to_grab
        self.days_predict = days_to_predict
        self.get = exchange + '/' + symbol
        self._get_stock()
        self._scale_and_fit()
        self._predict()
    
    def plot_hist(self):
        plt.plot(self.stock['Close'])
        plt.title(self.symbol+' Stock History')
        plt.xlabel('Date')
        plt.ylabel('Value (US$)')
        
    def _get_stock (self):
        if self.rows == -1:
            self.stock = quandl.get(self.get)

        else:
            self.stock = quandl.get(self.get,rows=self.rows)
    def _scale_and_fit (self):
        data = np.array(self.stock['Close'].values).reshape(-1,1)

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data)
        X_train = [] 
        y_train = []
        for i in range((self.days_predict*2),len(data)-self.days_predict):
            X_train.append(scaled_data[i-(self.days_predict*2):i,0])
            y_train.append(scaled_data[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
    
        model.compile(loss=root_mean_squared_error, optimizer='adam')
        model.fit(X_train, y_train, epochs=1, batch_size=32)
        self.model = model
    
    def _predict(self):
        data = self.stock['Close']
        predictions = data[-(self.days_predict*2):]
        for i in range (self.days_predict):
            x = np.array(predictions[-(self.days_predict*2):]).reshape(-1,1)
            scaled_x = self.scaler.fit_transform(x)
            scaled_x = scaled_x.reshape(1,-1,1)
            pred = self.model.predict(scaled_x)
            pred = self.scaler.inverse_transform(pred)
            predictions = predictions.append(pd.Series(pred[0][0]),ignore_index=True)
        df = pd.DataFrame()
        df['Points'] = predictions.values
        dates_index = pd.date_range(self.stock.index[len(self.stock)-self.days_predict],periods=(self.days_predict*3))
        df['dates'] = dates_index
        df = df.set_index('dates')
        #predictions = predictions.reindex(dates_index)
        self.predictions = df
        
    def show(self):
        hist = self.predictions[:self.days_predict*2]
        pred = self.predictions[-self.days_predict:]
        plt.plot(hist)
        plt.plot(pred)
        plt.legend(['History','Predictions'])
        plt.xticks(rotation=90)
        plt.xlabel('Date')
        plt.ylabel('Value (US$0)')
        plt.title(self.symbol + ' prediction')
        