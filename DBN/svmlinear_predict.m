clc;
clear;
addpath('D:/missdd/ML_Project/SVM/liblinear-2.11/matlab')
svm_data;
%[Y_train X_train] = libsvmread('G:\users\missdd\spyder_temp\LIBSVM\YearPredictionMSD');
%[Y_test X_test] = libsvmread('G:\users\missdd\spyder_temp\LIBSVM\YearPredictionMSD.t');
X_train = sparse(X_train);
X_test = sparse(X_test);
Y_predict = [];
acc = [];
t = [];
n_station = size(Y_train, 2);
for i = 1:n_station
    fprintf(1, 'station: %d', i);
    y_train = Y_train(:, i);
    y_test = Y_test(:, i);
    model = train(y_train, X_train, '-s 11 -e 1e-4 -p 0.01');
    tic;
    [y_predict, accuracy, decision_values] = predict(y_test, X_test, model);
    t = [t toc];
    Y_predict = [Y_predict y_predict];
    acc = [acc accuracy];
    fprintf(1, '\n');
end
MAPE = mean(mean(abs(Y_predict - Y_test)./Y_test))*100;
[acc_sort acc_index] = sort(acc(2,:));
fprintf(1, 'MAPE: %d', MAPE);
fprintf(1, '\n');
fprintf(1, 'time: %f', sum(t));
fprintf(1, '\n');
