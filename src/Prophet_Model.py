#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:22:47 2019

@author: seth
"""
import pandas as pd
import quandl
from fbprophet import Prophet
import matplotlib.pyplot as plt
import os
from suppress import suppress_stdout_stderr
import numpy as np

plt.style.use('fivethirtyeight')
quandl.ApiConfig.api_key = os.environ.get('quandl')

stock = quandl.get('WIKI/MSFT',rows=1305)

X = stock.index
y = stock.Close

X_test = X[1044:1305:]
y_test = y[1044:1305:]
X_train = X[:1044]
y_train = y[:1044]

def test_prph(X_train, X_test, y_train, y_test):
    train = pd.DataFrame()
    train['ds'] = list(X_train)
    train['y'] = list(y_train)
    test = pd.DataFrame()
    test['ds'] = list(X_test)
    test['y'] = list(y_test)
    
    with suppress_stdout_stderr():
        model = Prophet(
                            daily_seasonality=False,
                            weekly_seasonality=False,
                            yearly_seasonality=True,
                            changepoint_prior_scale=.05
                            )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(train)
        future = model.make_future_dataframe(periods=261)
        forecast = model.predict(future)
        return(forecast.yhat.tail(261))

def graph_forcast(forecast,y_train,y_test):
    test_val = pd.DataFrame()
    test_val['Actual'] = y_test
    test_val['Predicted'] = forecast.values
    
    plt.plot(y_train)
    plt.plot(test_val[['Actual','Predicted']])
    plt.legend(['History','Actual','Predict'])
    plt.title('Prophet Model Predictions 1 Year')

forecast = test_prph(X_train,X_test,y_train,y_test)
a = y_test.values
b = forecast.values
rmse = np.sqrt(np.mean((b-a)**2))

graph_forcast(forecast,y_train,y_test)
dat = pd.DataFrame()
dat['Actuals'] = a
dat['Predicted'] = b
