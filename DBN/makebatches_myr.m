load ./data/X_input_69_k2
load ./data/Y_output_69_k2
norm = max(max(X_input));
X_input = X_input/norm;
Y_output = Y_output/norm;

totnum=size(X_input,1);
rng('default'); %so we know the permutation of the training data
randomorder=randperm(totnum);
X_input = X_input(randomorder,:);
Y_output = Y_output(randomorder,:);

INPUT_train = X_input(1:14400,:);
INPUT_test = X_input(14401:17280,:);
Y_train = Y_output(1:14400,:);
Y_test = Y_output(14401:17280,:);

%train data process
totnum = size(INPUT_train,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);
numbatches = floor(totnum/144);
numdims  =  size(INPUT_train,2);
numdims_out = size(Y_train,2);
batchsize = 144;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, numdims_out, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = INPUT_train(1+(b-1)*batchsize:b*batchsize, :);
  batchtargets(:,:,b) = Y_train(1+(b-1)*batchsize:b*batchsize, :);
end;

%test data process
totnum=size(INPUT_test,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

numbatches = floor(totnum/144);
numdims  =  size(INPUT_test,2);
numdims_out = size(Y_test,2);
batchsize = 144;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, numdims_out, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = INPUT_test(1+(b-1)*batchsize:b*batchsize, :);
  testbatchtargets(:,:,b) = Y_test(1+(b-1)*batchsize:b*batchsize, :);
end;

rand('state',sum(100*clock)); 
randn('state',sum(100*clock));
