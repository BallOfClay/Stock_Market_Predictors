#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:18:34 2019

@author: seth
"""
import pandas as pd
import matplotlib.pyplot as plt
import quandl
from fbprophet import Prophet
from suppress import suppress_stdout_stderr
import os
plt.style.use('fivethirtyeight')
quandl.ApiConfig.api_key = os.environ.get('quandl')
class Stocks():
    
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

    def plot_history (self):
        plt.plot(self.stock.index,self.stock['Close'],color = 'g');
        plt.legend(['Cost of Stock (Closing Price)'],prop={'size':10})
        plt.title(self.symbol+' Stock History');
        plt.xlabel('Date');
        plt.ylabel('Price (US Dollars)');
        plt.grid(color = 'k', alpha = 0.4);
        plt.xticks(rotation=90);
