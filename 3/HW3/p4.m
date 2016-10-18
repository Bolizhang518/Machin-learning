%lasso regularization to get features

[digit text] = xlsread('ecs171.dataset.xlsx');
str_data = text(2:195,2:5);

beta = B(:, FitInfo.Index1SE);
x1 = x(:, beta ~= 0);
ystrain = str_data(:, 1);
ymedium = str_data(:, 2);
ystress = str_data(:, 3);
ygen = str_data(:, 4);

%% Linear kernel, default for two-class learning
[~, mse, area] = svm_binary(x1, ystrain, 'linear', true);
%%
[~, mse, area] = svm_binary(x1, ymedium, 'linear', true);
%%
[~, mse, area] = svm_binary(x1, ystress, 'linear', true);
%%
[~, mse, area] = svm_binary(x1, ygen, 'linear', true);
%% polyOrder to specify a polynomial kernel of order polyOrder.

[~, mse, area] = svm_binary(x1, ystrain, 'polynomial', true);
%%
[~, mse, area] = svm_binary(x1, ymedium, 'polynomial', true);
%%
[~, mse, area] = svm_binary(x1, ystress, 'polynomial', true);
%%
[~, mse, area] = svm_binary(x1, ygen, 'polynomial', true);
