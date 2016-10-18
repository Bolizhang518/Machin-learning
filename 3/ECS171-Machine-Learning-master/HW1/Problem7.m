%Problem7 perform prediction on a new data with three models

%Read in file(for convenience, we read all numbers with double type
%in this problem):
fid = fopen('D:/Rwd/auto-mpg.dat');
mpgdat = textscan(fid, '%f%f%f%f%f%f%f%f%q', 'TreatAsEmpty', '?');
fclose(fid);

%get out all columns except the car name and trasfrom into a matrix
mpgmat = cell2mat(mpgdat(1:8));
mpgmat = mpgmat(~any(isnan(mpgmat')), :);
trainset = mpgmat(1:280, :);
testset = mpgmat(281:end, :);

%newdata = [6, 300, 170, 3600, 9, 80, 1]; 

%second-order polynomial
%Since we have picked out the best features(horsepower and model year)
%We only make prediction based on these two features.

%horsepower prediction
beta11 = polyre(trainset(:, [1 4]), 2);
pred11 = beta11(1) + beta11(2) * 170 + beta11(3) * 170^2; %13.285, belongs to low mpg

%model year prediction
beta12 = polyre(trainset(:, [1 7]), 2);
pred12 = beta12(1) + beta12(2) * 80 + beta12(3) * 80^2; %26.8617, belongs to median mpg


%multi-variate polynomial
beta2 = polyremod(trainset);
newdata1 = [1, 6, 6^2, 300, 300^2, 170, 170^2, 3600, 3600^2, 9, 9^2, 80, 80^2, 1, 1];
pred2 = newdata1 *beta2; %20.1756 belongs to median mpg.

%logistic regression.
beta3 = stoc_grad_desc_logi(trainMat, zeros(1, 16), 0.1);

%p1 - p3 are probability of categorized in class low, median and high
newdata2 = [1, 6, 300, 170, 3600, 9, 80, 1]; 
exp1 = exp(newdata2*beta3(1:8)');
exp2 = exp(newdata2*beta3(9:16)');
p1 = exp1 / (exp1 + exp2 + 1);
p2 = exp2 / (exp1 + exp2 + 1);
p3 = 1 / (exp1 + exp2 + 1);
%p1 = 0.8538, p2 = 0.1462, p3 = 1.1343e-6, should belong to low mpg.
