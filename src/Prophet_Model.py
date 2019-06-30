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
