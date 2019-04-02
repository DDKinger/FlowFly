# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:56:08 2017

@author: missdd
"""
import scipy.io as sio 
def load_mat():
    File_Xinput = '/home/dd/DD/ML_Project/Traffic_flow/DBN/data/X_input_69_k6_h10'
    File_Youtput = '/home/dd/DD/ML_Project/Traffic_flow/DBN/data/Y_output_69_k6_h10'
    print("---- Dataset 69_k6_h10 ----")
    data1 = sio.loadmat(File_Xinput)
    data2 = sio.loadmat(File_Youtput)
    return data1['X_input'], data2['Y_output']