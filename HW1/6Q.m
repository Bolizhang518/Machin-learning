%% Load Data  
clc; clear;
%  contains the label. 
data = load('autompg.txt');

X = data(1:280,2:3);
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
options = optimset('GradObj', 'on', 'MaxIter', 1000);  
[theta, cost] = ...
    fminunc(@(t)(costFunction1(t, X, y)), initial_theta, options); 

fprintf('Cost at theta found by fminunc: %f\n', cost);  
fprintf('theta: \n');  
fprintf(' %f \n', theta);

plotDecisionBoundary(theta, X, y); 
hold on;
xlabel('mpg score')
ylabel('fetures score')  

legend('low', 'Not low') 
hold off;

%===============================predict 
X_t = data(280:392,2:3);

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
sum(prob==0)   %=15
sum (prob==1)  %=98

sum(y_t ==1)  %8
sum(y_t ==0)  %105 
fprintf('MSE is : %f\n',sum(y_t ==0) / sum(prob==0)); 
sum(y_t ==0) / sum(prob==0)

