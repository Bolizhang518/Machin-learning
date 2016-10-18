function [ theta] = normalEqn( x,y )
%NORMALEQN Summary of this function goes here
%   Detailed explanation goes here

theta = inv(X'* X)*X'*y;

end

