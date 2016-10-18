function [theta, J_history] = gradientDescent(x, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

thetaLen = length(theta);
tempVal = theta;
for iter = 1:num_iters
    temp = (x*theta - y);


 for i=1:thetaLen
        tempVal(i,1) = sum(temp.*x(:,i));
    end
    
    theta = theta - (alpha/m)*tempVal;
    
    J_history(iter,1) = computeCost(x,y,theta);
 
end