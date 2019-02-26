clc;
clear;
addpath('D:/missdd/ML_Project/SVM/libsvm-3.22/matlab')
svm_data;
Y_predict = [];
acc = [];
t = [];
n_station = size(Y_train, 2);
for i = 1:n_station
    fprintf(1, 'station: %d', i);
    y_train = Y_train(:, i);
    y_test = Y_test(:, i);
    model = svmtrain(y_train, X_train, '-s 3 -e 1e-4 -p 0.05');
    tic;
    [y_predict, accuracy, decision_values] = svmpredict(y_test, X_test, model);
    t = [t toc];
    Y_predict = [Y_predict y_predict];
    acc = [acc accuracy(2)];
    fprintf(1, '\n');
end
MAPE = mean(mean(abs(Y_predict - Y_test)./Y_test))*100;
MAE = mean(mean(abs(Y_predict - Y_test)))*norm;
RMSE = sqrt(mean(mean(abs(Y_predict - Y_test).^2)))*norm;
[acc_sort acc_index] = sort(acc);
fprintf(1, 'MAPE: %d', MAPE);
fprintf(1, '\n');
fprintf(1, 'MAE: %d', MAE);
fprintf(1, '\n');
fprintf(1, 'RMSE: %d', RMSE);
fprintf(1, '\n');
fprintf(1, 'time: %f', sum(t));
fprintf(1, '\n');
