# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:03:17 2017

@author: missdd
"""
import numpy as np
#import pickle
#from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import scipy.io as sio 


def load_mat():
    File_Xinput = '/home/dd/DD/ML_Project/Traffic_flow/DBN/data/X_input_69_k2'
    File_Youtput = '/home/dd/DD/ML_Project/Traffic_flow/DBN/data/Y_output_69_k2'
    data1 = sio.loadmat(File_Xinput)
    data2 = sio.loadmat(File_Youtput)
#    return data1['X_input_weekend'], data2['Y_output_weekend']
    return data1['X_input'], data2['Y_output']


X_input,Y_output = load_mat()
X_input = X_input.astype('float32')
Y_output = Y_output.astype('float32')

norm = np.amax(X_input)
X_input /= norm
X_train = X_input[0:14400,:]
X_test = X_input[14400:,:]
X_mean = np.mean(X_train, axis = 0)
X_std  = np.std(X_train, axis = 0)
X_train -= X_mean
X_train /= X_std
X_test -= X_mean
X_test /= X_std
Y_output /= norm
Y_train = Y_output[0:14400,:]
Y_test = Y_output[14400:,:]
#add sparsity
#X_train *= np.array(np.random.uniform(size = X_train.shape) > 0.3, dtype = 'float32')

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
print(Y_train.shape[1], 'OUTPUT')


x_input = Input(shape=(138,))
x = Dense(256, activation='sigmoid')(x_input)
x = Dropout(0.5)(x)
x = Dense(256, activation='sigmoid')(x)
x = Dropout(0.5)(x)
y_output = Dense(69, activation='sigmoid', name = 'y_out')(x)
model = Model(x_input, y_output)

model.summary()
#plot(model)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_percentage_error'])

history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    batch_size=144, nb_epoch=1000,
                    verbose=0, shuffle=True)
score = model.evaluate(X_test, Y_test, verbose=0)
Y_predict = model.predict(X_test)
MAE_test = np.mean(np.abs(Y_test - Y_predict))*norm
RMSE_test = np.sqrt(np.mean((Y_test - Y_predict)**2))*norm

print('Test score:', score[0])
print('MAPE_test:', score[1])
print('MAE_test:', MAE_test)
print('RMSE_test:', RMSE_test)