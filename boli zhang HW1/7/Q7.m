%% Clear and Close Figures

clear all; close all; clc




data = load('autompg.txt');
X = data(:,2:6);
y = data(:,1);
m = length(y);


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


% Choose some alpha value
alpha = 0.003;
num_iters = 1000;

% Init Theta and Run Gradient Descent 
theta = zeros(6, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 3);
xlabel('Number of iterations');
ylabel('predict MPG ');
title ( 'MPG rating that we should have expected')

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 6 cylinders, 300 cc displacement, 170 horsepower, 3600 lb weight,9 m/sec2 acceleration

x_predict = [1 6 300 170 3600 9];
for i=2:6
    x_predict(i) = (x_predict(i) - mu(i-1)) / sigma(i-1);
end
price = x_predict * theta;


% ============================================================

fprintf(['Predicted 6 cylinders, 300 cc displacement, 170 horsepower, 3600 lb weight,9 m/sec2 acceleration, ' ...
         '(using gradient descent):\n mpg %f\n'], price);

