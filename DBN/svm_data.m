load ./data/X_input_69_k2
load ./data/Y_output_69_k2
norm = max(max(X_input));
X_input = X_input/norm;
Y_output = Y_output/norm;

X_train = X_input(1:14400,:);
X_test = X_input(14401:end,:);

% Xmean      = mean(X_train);
% Xstd       = std(X_train);    
% X_train = bsxfun(@rdivide,bsxfun(@minus,X_train,Xmean),Xstd);
% X_test = bsxfun(@rdivide,bsxfun(@minus,X_test,Xmean),Xstd);

Y_train = Y_output(1:14400,:);
Y_test = Y_output(14401:end,:);

% random the order
% rng('default'); %so we know the permutation of the training data
% randomorder=randperm(totnum);
% X_train = X_train(randmorder, :);
% Y_train = Y_train(randmorder, :);

fprintf(1, 'Size of the training dataset= %5d \n', size(X_train, 1));
fprintf(1, 'Size of the training dataset= %5d \n', size(X_test, 1));