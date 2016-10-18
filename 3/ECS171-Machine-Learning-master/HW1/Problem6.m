%Problem6: Modified solver for multinomial logistic regression(1st order)

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

%Report training and test MSE
[~, mse] = stoc_grad_desc_logi(trainset, zeros(1, 16), 0.1, testset);
