# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:01:52 2017

@author: missdd
"""
import time
import matplotlib.pyplot as plt
import numpy as np

def xaxis_transfer(Y_test,Y_predict):
    x1 = time.strptime('2016-01-01 00:00:00','%Y-%m-%d %X')
    x2 = time.strptime('2016-01-02 00:00:00','%Y-%m-%d %X')
    x3 = time.mktime(x1)
    x4 = time.mktime(x2)
    delta = (x4 - x3) / (24*12)
    
    x0 = x3
    xt = [time.strftime('%H:%M',x1)]
    for i in range(24*12-1):
        x0 = x0 + delta
        x00 = time.localtime(x0)
        x00 = time.strftime('%H:%M',x00)
        xt.append(x00)
              
    x = np.arange(24*12)
    fig,ax = plt.subplots(figsize = (10,5))
    plt.plot(x,Y_test,'-k.',x,Y_predict,'-m.')
    plt.xticks(x, xt, rotation = 90)
    
    aticks = np.arange(0,len(xt),12)
    alables = [xt[index] for index in aticks]
    aticks = np.append(aticks, len(xt))
    alables.append('24:00')
    ax.set_xticks(aticks)
    ax.set_xticklabels(alables)