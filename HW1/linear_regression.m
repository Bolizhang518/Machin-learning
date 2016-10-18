function [ a0 a1 ] = linear_regression( x,y ) %input at the end

%LINEAR_REGRESSION Summary of this function goes here
%   Detailed explanation goes here
n = length(x);

a1 = (n*sum(x.*y)-sum(x)*sum(y))/(n*sum(x.^2) -(sum(x))^2);
a0 = mean(y)-a1*mean(x);


end

