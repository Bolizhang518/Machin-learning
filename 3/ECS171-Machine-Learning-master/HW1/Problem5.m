%Problem5 Modified function for 2nd order regression of all features(15 terms).

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
[~, mse, ~] = polyremod(trainset, testset);