%Problem4 Report MSE and plot regression lines.

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

%MSE matrix is 10 by 7 each column represents a feature, each
%feature has 10 rows. Each two rows represents the traing MSE and 
%test MSE for a polynomial order.

%e.g, for row6, column3, the number is the test MSE of 2 order polynomial
%regression model with the 'horsepower' fearure.
MSE = zeros(10, 7);
for i = 1:7
    for j = 1:5
        %Only mse is needed
        [~, mse, ~] = polyre(trainset(:, [1 (i+1)]), j - 1, testset(:, [1 (i+1)]));
        MSE([2*j - 1, 2*j], i) = mse; 
    end
end

%Plot
minval = min(testset);
maxval = max(testset); %Fetch max and min value for each feature
%Get the names for each feature for naming xlabel and title.
label = {'cylinders', 'displacement','horsepower',...
          'weight', 'acceleration', 'model year', 'origin'};
for i = 1:7
    plot(testset(:, i+1), testset(:, 1), 'ko') %plot data points
    hold on
    xlabel(label{i})
    ylabel('mpg')
    title(sprintf('plot for mpg vs %s for order 0-4', label{i}))
    color = {'b', 'r', 'y', 'g', 'c'};
    for j = 1:5
        x = linspace(minval(i + 1), maxval(i + 1), 5000);
        %Only prediction results are needed.
        [~, ~, pred] = polyre(trainset(:, [1 (i+1)]), j - 1, [x' x']);
        plot(x', pred, sprintf('%c-', color{j}))
        hold on
    end
    %create legend
    legend('data points', '0th order', '1th order', '2th order', '3th order', '4th order')
    hold off
    saveas(gcf, sprintf('plot%d', i), 'jpg')
    clf
end


