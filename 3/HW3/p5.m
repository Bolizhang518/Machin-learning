%linear kernel 
[digit text] = xlsread('ecs171.dataset.xlsx');
str_data = text(2:195,2:5);

beta = B(:, FitInfo.Index1SE);
x1 = x(:, beta ~= 0);
ystrain = str_data(:, 1);
ymedium = str_data(:, 2);
ystress = str_data(:, 3);
ygen = str_data(:, 4);
%%
[testclass1, ~] = svm_binary(x1,ymedium, 'linear', false);
[testclass2, ~] = svm_binary(x1, ygen, 'linear', false);
testclass = strcmp(testclass1, ymedium) & strcmp(testclass2, ygen);
MSE_separate = mean(~testclass); 

[~, mse_comp, area_composite] = svm_com(x1, str_data(:,[2,4]), 'linear', true); 