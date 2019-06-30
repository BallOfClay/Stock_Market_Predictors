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

quandl.ApiConfig.api_key = os.environ.get('quandl')

df = quandl.get('WIKI/MSFT',rows=1305)

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
    