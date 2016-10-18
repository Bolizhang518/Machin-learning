
%Compare result with problem5
%beta = B(:, FitInfo.Index1SE);
xnew = x(:, beta ~= 0);
[~,score,eigenval] = pca(xnew);%checked pca(x) but with MSE more than 80%, thus I decided to apply pca on xnew
ratio = sum(eigenval(1:3))/ sum(eigenval); %0.4667

prinx = score(:, 1:3);

ymedium = str_data(:, 2);
ystress = str_data(:, 4);
[testclass1_pca, ~, area_medium_pca] = svm_com(prinx,ymedium,'linear', true);
[testclass2_pca, ~, area_perturb_pca] = svm_com(prinx,ystress, 'linear', true);
testclass_pca = strcmp(testclass1_pca, ymedium) & strcmp(testclass2_pca, ystress);
MSE_separate_pca = mean(~testclass_pca);

[~, MSE_composite_pca, area_composite_pca] = svm_com(prinx, str_data(:,[2,4]), 'linear', true); 

