load ./data/X_input_69_k2
load ./data/Y_output_69_k2
norm = max(max(X_input));
X_input = X_input/norm;
Y_output = Y_output/norm;

fprintf(1, '------GMAKE------\n');
fprintf(1, 'load k2 \n');
fprintf(1, 'Norm = %d \n', norm);
fprintf(1, 'Size of X_input= %d, %d \n', size(X_input));
fprintf(1, 'Size of Y_output= %d, %d \n', size(Y_output));

%add Gaussion noise
% noise_factor = 0.1;
% X_input_noisy = X_input + noise_factor * randn(size(X_input));
% X_input_noisy = max(min(X_input_noisy,1),0);
% INPUT_train_noisy = X_input_noisy(1:14400,:);
% INPUT_test_noisy = X_input_noisy(14401:17280,:);

n_total = length(X_input);
n_train = n_total/12*10
INPUT_train = X_input(1:n_train,:);
INPUT_test = X_input(n_train+1:end,:);

Xmean      = mean(INPUT_train);
Xstd       = std(INPUT_train);    
INPUT_train = bsxfun(@rdivide,bsxfun(@minus,INPUT_train,Xmean),Xstd);
INPUT_test = bsxfun(@rdivide,bsxfun(@minus,INPUT_test,Xmean),Xstd);

Y_train = Y_output(1:n_train,:);
Y_test = Y_output(n_train+1:end,:);

%train data process
totnum=size(INPUT_train,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rng('default'); %so we know the permutation of the training data
randomorder=randperm(totnum);

batchsize = 144;
numbatches = floor(totnum/batchsize);
numdims  =  size(INPUT_train,2);
numdims_out = size(Y_train,2);
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, numdims_out, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = INPUT_train(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = Y_train(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;

%test data process
totnum=size(INPUT_test,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

% rand('state',0); %so we know the permutation of the training data
% randomorder=randperm(totnum);

batchsize = 144;
numbatches = floor(totnum/batchsize);
numdims  =  size(INPUT_test,2);
numdims_out = size(Y_test,2);
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, numdims_out, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = INPUT_test(1+(b-1)*batchsize:b*batchsize, :);
  testbatchtargets(:,:,b) = Y_test(1+(b-1)*batchsize:b*batchsize, :);
end;

rand('state',sum(100*clock)); 
randn('state',sum(100*clock));
