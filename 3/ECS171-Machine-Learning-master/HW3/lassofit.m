%Self written lasso
%We provide a lambda range to find the best value, may be tested for many times. 
function [b, fitinfo] = lassofit(x, y, lambda)
%lambda      the lambda range to test, an array 
%b           coefficients
%fitinfo     a struct containing MSE, Lambda with minimum MSE, index of lambda

%All data are already in the same magnitude thus we only need to center the data.
xnew = x - repmat(mean(x), size(x, 1), 1);
ynew = y - mean(y);
[b, MSE] = CV_lasso(xnew, ynew, lambda);
[~, I] = min(MSE);
fitinfo = struct('lambda_val', lambda(I), 'lambda_ind', I, 'minMSE', MSE(I)); 

end

function [beta, MSE] = CV_lasso(x, y, lambda)
%Assume x, y have been centered, use 10 folds CV, use proximal gradient descent to find beta
    ind = floor(linspace(1, length(y) + 1, 11)); %[a,b)
    MSE = zeros(1, length(lambda));
    for i = 1:10
        testx = x(ind(i):(ind(i+1) -1), :);
        testy = y(ind(i):(ind(i+1) -1));
        trainx = x; trainy = y;
        trainx(ind(i):(ind(i+1) -1), :) = [];
        trainy(ind(i):(ind(i+1) -1)) = [];
        
        b = prox_grad(trainx, trainy, lambda);
        
        MSE = MSE + sum((repmat(testy, 1, length(lambda)) - testx * b).^2); 
    end
    MSE = MSE ./ size(x, 1);
    [~, I] = min(MSE);
    beta = prox_grad(x, y, lambda(I));
end

function beta = prox_grad(x, y, lambda)
%proximal gradient descent
    beta = repmat(rand(size(x, 2), 1) - 0.5, 1, length(lambda));
    for j = 1:10000
        grad_cont = x' * (x * beta - repmat(y, 1, length(lambda)));
        a = beta - 0.00001 .* grad_cont;
        z = repmat(lambda, size(x, 2), 1) .* 0.00001;
        beta1 = a - z; beta2 = a + z;
        beta1(a <= z) = 0;
        beta2(a >= -z) = 0;
        beta = beta1 + beta2;
    end
end