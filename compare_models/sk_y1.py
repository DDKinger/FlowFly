# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:37:29 2017

@author: missdd
"""

from LoadTrafficData import load_mat
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xaxis_transfer import xaxis_transfer
import numpy as np
import time

X_input,Y_output = load_mat()
X_input = X_input.astype('float64')
Y_output = Y_output.astype('float64')

norm = np.amax(X_input)
X_input /= norm
X_train = X_input[:14400,:]
X_test = X_input[14400:,:]
X_mean = np.mean(X_train, axis = 0)
X_std  = np.std(X_train, axis = 0)
# X_train -= X_mean
# X_train /= X_std
# X_test -= X_mean
# X_test /= X_std
Y_output /= norm
Y_train = Y_output[:14400,:]
Y_test = Y_output[14400:,:]
y_pred = np.zeros([Y_test.shape[0], Y_test.shape[1]])

print("X_train data type:", X_train.dtype, "  X_train shape:", X_train.shape)
print("Y_train data type:", Y_train.dtype, "  Y_train shape:", Y_train.shape)

method = [
          'DecisionTreeRegressor(max_depth = 10, random_state=0)',
        #   'LinearRegression()', 
          'KNeighborsRegressor()',
          'RandomForestRegressor(random_state=0)',
          'MLPRegressor(random_state=0, hidden_layer_sizes=(512,),activation="logistic",batch_size=144)'
         ]
for s in method:
    t2 = 0
    print("Start training", s[:s.find('(')])
    clf = eval(s)
    print(clf)
    clf.fit(X_train,Y_train)
    t1 = time.clock()
    y_pred = clf.predict(X_test)
    t2 = time.clock() - t1
    y_true = Y_test
    diff = np.abs(Y_test - y_pred) / y_true
    mape = 100. * np.mean(diff)
    mape_s = 100.*np.mean(diff, axis = 0)
    mape_s_sort = np.argsort(mape_s)
    mae = np.mean(np.abs(y_true - y_pred))*norm
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))*norm
    print('MAPE:', mape, '%')
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('prediction time: %.3f S'%t2)
    print('')
    
#    xaxis_transfer(y_true[0:24*12,6]*norm,y_pred[0:24*12,6]*norm)
#    with open('tree.dot', 'w') as f:
#         f = export_graphviz(clf, out_file=f)

#import pydotplus 
#dot_data = export_graphviz(clf, out_file=None) 
#graph = pydotplus.graph_from_dot_data(dot_data) 
#graph.write_pdf("tree2.pdf")