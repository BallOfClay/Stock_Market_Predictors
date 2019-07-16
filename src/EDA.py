#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 09:56:23 2019

@author: seth
"""
import os
import quandl
quandl.ApiConfig.api_key = os.environ.get('quandl')

msft = quandl.get('WIKI/MSFT',rows=1305)
aapl = quandl.get('WIKI/AAPL',rows=1305)
jnj = quandl.get('WIKI/JNJ',rows=1305)

