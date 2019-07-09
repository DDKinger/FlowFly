% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

clear all
close all

% diary('G:\MachineLearning\Traffic Flow Prediction2\prd log.txt');
% diary on;
t1 = clock; %��ʼ��ʱ
maxepoch=20;
numhid=256; numpen=256; %numpen2=256; numpen3=256; numpen4=256; numpen5=256; 
fprintf(1,'Pretraining a deep autoencoder. \n');

gmakebatches_my;
numoutput = size(Y_output,2);
[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
grbm;
hidrecbiases=hidbiases; 
restruct_error1 = restruct_error;
save mnistvhclassify vishid hidrecbiases visbiases restruct_error1;

% fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
% batchdata=batchposhidprobs;
% numhid=numpen;
% restart=1;
% rbm;
% hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
% restruct_error2 = restruct_error;
% save mnisthpclassify hidpen penrecbiases hidgenbiases restruct_error2;
 
% fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
% batchdata=batchposhidprobs;
% numhid=numpen2;
% restart=1;
% rbm;
% hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
% restruct_error3 = restruct_error;
% save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2 restruct_error3;
% 
% fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen,numpen2);
% batchdata=batchposhidprobs;
% numhid=numpen3;
% restart=1;
% rbm;
% hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;
% restruct_error4 = restruct_error;
% save mnisthp3classify hidpen3 penrecbiases3 hidgenbiases3 restruct_error4;
% 
% fprintf(1,'\nPretraining Layer 5 with RBM: %d-%d \n',numpen,numpen2);
% batchdata=batchposhidprobs;
% numhid=numpen4;
% restart=1;
% rbm;
% hidpen4=vishid; penrecbiases4=hidbiases; hidgenbiases4=visbiases;
% restruct_error5 = restruct_error;
% save mnisthp4classify hidpen4 penrecbiases4 hidgenbiases4 restruct_error5;
% 
% fprintf(1,'\nPretraining Layer 6 with RBM: %d-%d \n',numpen,numpen2);
% batchdata=batchposhidprobs;
% numhid=numpen5;
% restart=1;
% rbm;
% hidpen5=vishid; penrecbiases5=hidbiases; hidgenbiases5=visbiases;
% restruct_error6 = restruct_error;
% save mnisthp5classify hidpen5 penrecbiases5 hidgenbiases5 restruct_error6;

backpropclassify1L; 
time_tol = etime(clock, t1); %������ʱ
fprintf(1,'Total time: %f\n', time_tol);
% imageshow;
% diary off;