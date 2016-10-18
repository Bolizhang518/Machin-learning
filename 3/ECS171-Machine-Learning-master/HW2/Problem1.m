%Read in the data
fid = fopen('D:/Rwd/yeast.data');
data = textscan(fid, '%q%f%f%f%f%f%f%f%f%q');
fclose(fid);

%get the numeric part data and combines with the class indicator matrix y.
yeast = cell2mat(data(2:9));
y = zeros(size(yeast, 1), 10);
class = {'CYT', 'NUC', 'MIT', 'ME3', 'ME2', 'ME1', 'EXC', 'VAC', 'POX', 'ERL'};
datclass = cellstr(data{10});
for i = 1:10
    y(:, i) = double(cell2mat(cellfun(@(x) strcmp(x, class{i}), datclass, 'uniformOutput', false)));
end
yeast = [y, yeast];

%Permute matrix for stochastic gradient descent.
rng(171);
perm = randperm(size(yeast, 1));
permyeast = yeast(perm, :);

%Split into training data and test data
trainMat = permyeast(1:round(0.65*size(yeast, 1)), :);
testMat = permyeast((round(0.65*size(yeast, 1)) + 1):end, :);

[beta, trainerr, trainout4, trainhid2, trainw42, testerr, testout4, testhid2] = ANN_new(trainMat, 5000, 0.1, testMat);

%Plot Error change for training data and testing data
plot(trainerr, 'b-')
hold on;
plot(testerr, 'r-')
xlabel('Iteration')
ylabel('Error Change')
title('Training and testing error change')
legend('Training Error', 'Testing Error')

%Plot 4th output node and 2nd hidden node change
plot(trainout4, 'b-')
hold on;
plot(testout4, 'b--')
hold on;
plot(trainhid2, 'r-')
hold on;
plot(testhid2, 'r--')

xlabel('Iteration')
ylabel('output and hidden node change')
title('4h output node and 2nd hidden node change')

legend('4th output node training change', '4th output node test change', ...
    '2th hidden node training change', '2th hidden node test change')

%Plot w42 weight change
plot(trainw42)
xlabel('Iteration')
ylabel('w42 change')
title('weight change between 4th output node and 2nd hidden node')
