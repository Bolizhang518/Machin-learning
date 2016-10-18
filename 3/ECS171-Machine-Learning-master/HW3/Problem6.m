%Problem6
%Compare result with problem5
beta = b0(:, fitinfo0.Index1SE);
xnew = x(:, beta ~= 0);
[~,score,eigenval] = pca(xnew);%checked pca(x) but with MSE more than 80%, thus I decided to apply pca on xnew
ratio = sum(eigenval(1:3))/ sum(eigenval); %0.4667

prinx = score(:, 1:3);

ynew2 = str_data(:, 3);
ynew4 = str_data(:, 5);
[testclass1_pca, ~, area_medium_pca] = SVM_multiclass_pkg(prinx,ynew2,'linear', true);
[testclass2_pca, ~, area_perturb_pca] = SVM_multiclass_pkg(prinx,ynew4, 'linear', true);
testclass_pca = strcmp(testclass1_pca, ynew2) & strcmp(testclass2_pca, ynew4);
MSE_separate_pca = mean(~testclass_pca);

[~, MSE_composite_pca, area_composite_pca] = SVM_composite_pkg(prinx, str_data(:,[3,5]), 'linear', true); 

