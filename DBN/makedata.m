clc;
clear;
load ./data/X_input_70_k5
load ./data/Y_output_70_k5
k=5;
X_input(:,[59*k+1:60*k])=[];
Y_output(:,[60])=[];
%save X_input_68_k4 X_input;
%save Y_output_68_k4 Y_output;