%% =================6 low mean squared error ====================================
%% Load Data  

%  contains the label. 
data = load('');

X = data();
y = data();
m = length(y);
%Part 2: Compute Cost and Gradient
%  Setup the data matrix appropriately 
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
 

% ============= Part 3: Optimizing   =============

%  Set options for fminunc  
options = optimset('GradObj', 'on', 'MaxIter',1000);  
[theta, cost] = ...
    fminunc(@(t)(costFunction1(t, X, y)), initial_theta, options); 
% ===============================predict 
X_t = data();
X_tr= data();

y_t = data();
y_t()=0
y_t()=1

y_tr = data();
y_tr()=0
y_tr()=1
%  Predict probability ,make feature line regression to [0.1]

prob = sigmoid([ones(113,1) X_t] * theta);

probtr = sigmoid([ones(280,1) X_tr] * theta);





