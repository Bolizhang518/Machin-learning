8%% ==================Q1=========================%%
data = load('autompg.txt');
mpg = data(:,1);
cylinders = data(:,2);
displacement = data(:,3);
horsepower = data(:,4);
weight = data(:,5);
accelration = data(:,6);
model  = data(:,7);
origin = data(:,8);

sortmpg = sort(mpg);
length_mpg = length(sortmpg);
threshold1 = (length_mpg)/3;
threshold2 = 2*((length_mpg)/3);

mpgthreshold1 = sortmpg(fix(threshold1),1);
mpgthreshold2 = sortmpg(fix(threshold2),1);
mpgthreshold1
mpgthreshold3
%% =================Q2===========================
x = [cylinders,displacement,horsepower,weight,accelration,model,origin];
varNames  = {'cylinders';'displacement';'horsepower';'weight';'accelration';'model';'origin'};
figure
gplotmatrix(x,[],mpg,[ 'c' 'b' 'm' 'g' 'y' 'r' 'r'],[],[],false);
%text([.08 .24 .43 .66 .83], repmat(-.1,1,5), varNames, 'FontSize',8);
%text(repmat(-.12,1,5), [.86 .62 .41 .25 .02], varNames, 'FontSize',8, 'Rotation',90);

%% ========================Q3===========================



%% =================6 low mean squared error ====================================
%% Load Data  
clc; clear;
%  contains the label. 
data = load('autompg.txt');

X = data(1:280,2:8);
y = data(1:280,1);
 y(y<19)=0
 y(y>18.5)=1
m = length(y);

% ==================== Part 1: Plotting ====================  
plotData1(X, y);  
% Put some labels  
hold on;  
% Labels and Legend  
xlabel('features')
ylabel('MPG')  

% Specified in plot order  

legend('low', 'Not low')  
hold off;
% ============ Part 2: Compute Cost and Gradient ============  
%  Setup the data matrix appropriately, and add ones for the intercept term  
[m, n] = size(X); 

% Add intercept term to x and X_test  
X = [ones(m, 1) X];

% Initialize fitting parameters  
initial_theta = zeros(n + 1, 1); 

% Compute and display initial cost and gradient  
[cost, grad] = costFunction1(initial_theta, X, y);  

fprintf('Cost at initial theta (zeros): %f\n', cost);  
fprintf('Gradient at initial theta (zeros): \n');  
fprintf(' %f \n', grad);  
 

% ============= Part 3: Optimizing using fminunc  =============

%  Set options for fminunc  
options = optimset('GradObj', 'on', 'MaxIter',1000);  
[theta, cost] = ...
    fminunc(@(t)(costFunction1(t, X, y)), initial_theta, options); 

%fprintf('Cost at theta found by fminunc: %f\n', cost);  
%fprintf('theta: \n');  
%fprintf(' %f \n', theta);

%plotDecisionBoundary(theta, X, y); 
%hold on;
%xlabel('mpg score')
%ylabel('fetures score')  

%legend('low', 'Not low') 
%hold off;

% ===============================predict 
X_t = data(280:392,2:8);

y_t = data(280:392,1);
y_t(y_t<19)=0
y_t(y_t>18.5)=1

%  Predict probability ,make feature line regression to [0.1]

prob = sigmoid([ones(113,1) X_t] * theta);


% p = predict(theta,X);
fprintf(['For testing data(), we predict an not low mpg ' ...
    'probability of %f\n\n'], prob);

% fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100); 

prob(prob < 0.5 )=0
prob(prob > 0.5 )=1
sum(prob==0)   %=9
sum (prob==1)  %=104

sum(y_t ==1)  %105
sum(y_t ==0)  %8
fprintf('MSE is : %f\n',[sum(prob==0)-[(sum(prob==0)+sum(y_t==0))/2]]^2); 
%[sum(prob==0)-[(sum(prob==0)+sum(y_t==0))/2]]^2



%% ============================6 med mpg error ==================
%% Load Data  
clc; clear;
%  contains the label. 
data = load('autompg.txt');

X = data(1:280,2:8);
y = data(1:280,1);
 y(y>26.8)=1
 y(y<19)=1
 y(y>1)=0
m = length(y);

% ==================== Part 1: Plotting ====================  
plotData1(X, y);  
% Put some labels  
hold on;  
% Labels and Legend  
xlabel('features')
ylabel('MPG')  

% Specified in plot order  

legend('low', 'Not low')  
hold off;
% ============ Part 2: Compute Cost and Gradient ============  
%  Setup the data matrix appropriately, and add ones for the intercept term  
[m, n] = size(X); 

% Add intercept term to x and X_test  
X = [ones(m, 1) X];

% Initialize fitting parameters  
initial_theta = zeros(n + 1, 1); 

% Compute and display initial cost and gradient  
[cost, grad] = costFunction1(initial_theta, X, y);  

fprintf('Cost at initial theta (zeros): %f\n', cost);  
fprintf('Gradient at initial theta (zeros): \n');  
fprintf(' %f \n', grad);  
 

% ============= Part 3: Optimizing using fminunc  =============

%  Set options for fminunc  
options = optimset('GradObj', 'on', 'MaxIter',1000);  
[theta, cost] = ...
    fminunc(@(t)(costFunction1(t, X, y)), initial_theta, options); 

%fprintf('Cost at theta found by fminunc: %f\n', cost);  
%fprintf('theta: \n');  
%fprintf(' %f \n', theta);

%plotDecisionBoundary(theta, X, y); 
%hold on;
%xlabel('mpg score')
%ylabel('fetures score')  

%legend('low', 'Not low') 
%hold off;

% ===============================predict 
X_t = data(280:392,2:8);

y_t = data(280:392,1);


y_t(y_t>26.8)=1
 y_t(y_t<19)=1
 y_t(y_t>1)=0   %25
 
%  Predict probability ,make feature line regression to [0.1]

prob = sigmoid([ones(113,1) X_t] * theta);


% p = predict(theta,X);
fprintf(['For testing data(), we predict an not low mpg ' ...
    'probability of %f\n\n'], prob);

% fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100); 

prob(prob < 0.5 )=0
prob(prob > 0.5 )=1
sum(prob==0)   %=53
sum (prob==1)  %=60

sum(y_t ==1)  %88
sum(y_t ==0)  %25
fprintf('MSE is : %f\n',[sum(prob==0)-[(sum(prob==0)+sum(y_t==0))/2]]^2); 
%[sum(prob==0)-[(sum(prob==0)+sum(y_t==0))/2]]^2
%% ==============================6 High MPG ======================
clc; clear;
%  contains the label. 
data = load('autompg.txt');

X = data(1:280,2:8);
y = data(1:280,1);
 y(y<26.8)=1
 y(y>26.8)=0
m = length(y);


plotData1(X, y);  
% Put some labels  
hold on;  
% Labels and Legend  
xlabel('features')
ylabel('MPG')  

% Specified in plot order  

legend('low', 'Not low')  
hold off;
% ============ Part 2: Compute Cost and Gradient ============  
%  Setup the data matrix appropriately, and add ones for the intercept term  
[m, n] = size(X); 

% Add intercept term to x and X_test  
X = [ones(m, 1) X];

% Initialize fitting parameters  
initial_theta = zeros(n + 1, 1); 

% Compute and display initial cost and gradient  
[cost, grad] = costFunction1(initial_theta, X, y);  

fprintf('Cost at initial theta (zeros): %f\n', cost);  
fprintf('Gradient at initial theta (zeros): \n');  
fprintf(' %f \n', grad);  
 

% ============= Part 3: Optimizing using fminunc  =============

%  Set options for fminunc  
options = optimset('GradObj', 'on', 'MaxIter',1000);  
[theta, cost] = ...
    fminunc(@(t)(costFunction1(t, X, y)), initial_theta, options); 

%fprintf('Cost at theta found by fminunc: %f\n', cost);  
%fprintf('theta: \n');  
%fprintf(' %f \n', theta);

%plotDecisionBoundary(theta, X, y); 
%hold on;
%xlabel('mpg score')
%ylabel('fetures score')  

%legend('low', 'Not low') 
%hold off;

% ===============================predict 
X_t = data(280:392,2:8);

y_t = data(280:392,1);


y_t(y_t<26.8)=1
 
 y_t(y_t>26.8)=0   %25
 
%  Predict probability ,make feature line regression to [0.1]

prob = sigmoid([ones(113,1) X_t] * theta);


% p = predict(theta,X);
fprintf(['For testing data(), we predict an not low mpg ' ...
    'probability of %f\n\n'], prob);

% fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100); 

prob(prob < 0.5 )=0
prob(prob > 0.5 )=1
sum(prob==0)   %=51
sum (prob==1)  %=62

sum(y_t ==1)  %32
sum(y_t ==0)  %80
fprintf('MSE is : %f\n',[sum(prob==0)-[(sum(prob==0)+sum(y_t==0))/2]]^2); 
%[sum(prob==0)-[(sum(prob==0)+sum(y_t==0))/2]]^2




%% ===============================7================================

%% Clear and Close Figures

clear all; close all; clc

fprintf('Loading data ...\n');


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

fprintf('Program paused. Press enter to continue.\n');