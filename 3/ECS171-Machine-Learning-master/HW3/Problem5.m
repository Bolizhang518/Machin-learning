%Problem5
%Since linear kernel is the best according to mse, roc and prc curves, we only implement linear kernel for the following problems
beta = b0(:, fitinfo0.Index1SE);
xnew = x(:, beta ~= 0);
ynew2 = str_data(:, 3);
ynew4 = str_data(:, 5);
[testclass1, ~] = SVM_multiclass_pkg(xnew,ynew2, 'linear', false);
[testclass2, ~] = SVM_multiclass_pkg(xnew, ynew4, 'linear', false);
testclass = strcmp(testclass1, ynew2) & strcmp(testclass2, ynew4); %only two features both correct can be counted as correct prediction
MSE_separate = mean(~testclass); 

[~, MSE_composite, area_composite] = SVM_composite_pkg(xnew, str_data(:,[3,5]), 'linear', true); 