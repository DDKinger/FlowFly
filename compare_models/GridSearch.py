# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:21:46 2017

@author: DD
"""
from LoadTrafficData import load_mat
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

X_input,Y_output = load_mat()
X_input = X_input.astype('float64')
Y_output = Y_output.astype('float64')
norm = np.amax(X_input)
X_input /= norm
X_train = X_input[:14400,:]
X_test = X_input[14400:,:]
Y_output /= norm
Y_train = Y_output[:14400,:]
Y_test = Y_output[14400:,:]

#def mean_absolute_percent_error(y_true, y_pred, 
#                       sample_weight=None, 
#                       multioutput='uniform_average'):     
#
#    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput) 
#    output_errors = np.average((y_true - y_pred) ** 2, axis=0, weights=sample_weight)      
#    if isinstance(multioutput, string_types): 
#        if multioutput == 'raw_values': 
#            return output_errors 
#        elif multioutput == 'uniform_average': 
#            multioutput = None 
#    return np.average(output_errors, weights=multioutput)


tuned_parameters = [{'max_depth': [None, 10, 50, 100],
                     'max_features': ['auto', 'sqrt', 'log2']}] 

score = 'neg_mean_squared_error'
clf_g = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5, scoring=score) 
clf_g.fit(X_train, Y_train)

print("Best parameters set found on development set:")
print(clf_g.best_estimator_)
print("Grid scores on development set:")
re = clf_g.cv_results_
for (params, mean_score, rank) in zip(re['params'], re['mean_test_score'], re['rank_test_score']):
    print("%d (%0.3f) for %r"
        % (rank, mean_score, params))
y_true, y_pred = Y_test, clf_g.predict(X_test)