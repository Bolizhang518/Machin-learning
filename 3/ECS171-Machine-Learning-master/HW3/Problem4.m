%Problem4
%First use lasso regularization result to pick up features
beta = b0(:, fitinfo0.Index1SE);
xnew = x(:, beta ~= 0);
ynew1 = str_data(:, 2);
ynew2 = str_data(:, 3);
ynew3 = str_data(:, 4);
ynew4 = str_data(:, 5);

%get error rate for all features and draw ROC, PRC plots.
[~, mse11, area11] = SVM_multiclass_pkg(xnew, ynew1, 'linear', true);
[~, mse12, area12] = SVM_multiclass_pkg(xnew, ynew2, 'linear', true);
[~, mse13, area13] = SVM_multiclass_pkg(xnew, ynew3, 'linear', true);
[~, mse14, area14] = SVM_multiclass_pkg(xnew, ynew4, 'linear', true);

[~, mse21, area21] = SVM_multiclass_pkg(xnew, ynew1, 'polynomial', true);
[~, mse22, area22] = SVM_multiclass_pkg(xnew, ynew2, 'polynomial', true);
[~, mse23, area23] = SVM_multiclass_pkg(xnew, ynew3, 'polynomial', true);
[~, mse24, area24] = SVM_multiclass_pkg(xnew, ynew4, 'polynomial', true);

[~, mse31, area31] = SVM_multiclass_pkg(xnew, ynew1, 'RBF', true);
[~, mse32, area32] = SVM_multiclass_pkg(xnew, ynew2, 'RBF', true);
[~, mse33, area33] = SVM_multiclass_pkg(xnew, ynew3, 'RBF', true);
[~, mse34, area34] = SVM_multiclass_pkg(xnew, ynew4, 'RBF', true);