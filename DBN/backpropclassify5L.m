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

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.  

maxepoch=200;
fprintf(1,'\nFine Tune model. \n');
% fprintf(1,'60 batches of 1000 cases each. \n');

load mnistvhclassify
load mnisthpclassify
load mnisthp2classify
load mnisthp3classify
load mnisthp4classify

gmakebatches_my;
[numcases numdims numbatches]=size(batchdata);
N=numcases; 

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases];
w2=[hidpen; penrecbiases];
w3=[hidpen2; penrecbiases2];
w4=[hidpen3; penrecbiases3];
w5=[hidpen4; penrecbiases4];
w_pre = 0.1*randn(size(w5,2)+1,numoutput);
 

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w5,1)-1;
l6=size(w_pre,1)-1;
l7=numoutput; 
MAPE_train = [];
MAPE_test = [];
MAE_train = [];
MAE_test = [];
RMSE_train = [];   
RMSE_test = [];
Y_predict = [];
MAPE_station = zeros(1,numoutput);

for epoch = 1:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err_cr=0;
err_cr_a=0;
err_cr_s=0;
[numcases numdims numbatches]=size(batchdata);
N=numcases;
 for batch = 1:numbatches
  data = [batchdata(:,:,batch)];
  target = [batchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
  targetout = 1./(1 + exp(-w5probs*w_pre));

  err_cr = err_cr + mean(mean(abs(targetout - target)./target));
  err_cr_a = err_cr_a + mean(mean(abs(targetout - target)));
  err_cr_s = err_cr_s + sqrt(mean(mean(abs(targetout - target).^2)));
%   err_cr = err_cr + sum(sum(abs(targetout - target).^2/2))/N;
 end
 MAPE_train(epoch) = err_cr/numbatches*100;
 MAE_train(epoch) = err_cr_a/numbatches*norm;
 RMSE_train(epoch) = err_cr_s/numbatches*norm;
%%%%%%%%%%%%%% END OF COMPUTING TRAINING ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% COMPUTE TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
err_cr=0;
err_cr_a=0;
err_cr_s=0;
[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  target = [testbatchtargets(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
  w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
  targetout = 1./(1 + exp(-w5probs*w_pre));
  
  if epoch == maxepoch
      Y_predict = [Y_predict;targetout];
      MAPE_station = MAPE_station + mean(abs(targetout - target)./target);
  end

  err_cr = err_cr + mean(mean(abs(targetout - target)./target));
  err_cr_a = err_cr_a + mean(mean(abs(targetout - target)));
  err_cr_s = err_cr_s + sqrt(mean(mean(abs(targetout - target).^2)));
end
 MAPE_test(epoch) = err_cr/testnumbatches*100;
 MAE_test(epoch) = err_cr_a/testnumbatches*norm;
 RMSE_test(epoch) = err_cr_s/testnumbatches*norm;
 fprintf(1,'Before epoch %d Train # MAPE_train: %d . Test # MAPE_test: %d \t \t \n',...
            epoch,MAPE_train(epoch),MAPE_test(epoch));

%%%%%%%%%%%%%% END OF COMPUTING TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches/10
 fprintf(1,'epoch %d batch %d\r',epoch,batch);

%%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tt=tt+1; 
 data=[];
 targets=[]; 
 for kk=1:10
  data=[data 
        batchdata(:,:,(tt-1)*10+kk)]; 
  targets=[targets
        batchtargets(:,:,(tt-1)*10+kk)];
 end 

%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  max_iter=3;

  if epoch<6  % First update top-level weights holding other weights fixed. 
    N = size(data,1);
    XX = [data ones(N,1)];
    w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
    w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
    w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
    w4probs = 1./(1 + exp(-w3probs*w4)); w4probs = [w4probs  ones(N,1)];
    w5probs = 1./(1 + exp(-w4probs*w5)); %w5probs = [w3probs  ones(N,1)];

    VV = [w_pre(:)']';
    Dim = [l6; l7];
    [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,w5probs,targets);
    w_pre = reshape(X,l6+1,l7);

  else
    VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w_pre(:)']';
    Dim = [l1; l2; l3; l4; l5; l6; l7];
    [X, fX] = minimize(VV,'CG_CLASSIFY5L',max_iter,Dim,data,targets);

    w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
    xxx = (l1+1)*l2;
    w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
    xxx = xxx+(l2+1)*l3;
    w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
    xxx = xxx+(l3+1)*l4;
    w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
    xxx = xxx+(l4+1)*l5;
    w5 = reshape(X(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
    xxx = xxx+(l5+1)*l6;
    w_pre = reshape(X(xxx+1:xxx+(l6+1)*l7),l6+1,l7);

  end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 end

 save mnistclassify_weights w1 w2 w3 w4 w5 w_pre
 save mnistclassify_error MAPE_train MAPE_test MAE_train MAE_test RMSE_train RMSE_test;

end

fprintf(1,'MAPE_train:%d, MAPE_test:%d\n',MAPE_train(end),MAPE_test(end));
fprintf(1,'MAE_train:%d, MAE_test:%d\n',MAE_train(end),MAE_test(end));
fprintf(1,'RMSE_train:%d, RMSE_test:%d\n',RMSE_train(end),RMSE_test(end));
save Y_predict Y_predict;
MAPE_station = MAPE_station / testnumbatches * 100;
%save MAPE_station MAPE_station;
[MAPE_station_sort index] = sort(MAPE_station);
save MAPE_station_sort MAPE_station_sort index;



