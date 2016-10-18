function [ jVal,gradient ] = costFunction( theta )
%COSTFUNCTION2 Summary of this function goes here
%   Detailed explanation goes here
data = load('autompg.txt');
mpg = data(:,1);
cylinders = data(:,2);
displacement = data(:,3);
horsepower = data(:,4);
weight = data(:,5);
accelration = data(:,6);
model  = data(:,7);
origin = data(:,8);

x = mpg;
y = cylinders;
m=size(x,1);

hypothesis = h_func(x,theta);
delta = hypothesis - y;
jVal=sum(delta.^2);
gradient(1)=sum(delta)/m;
gradient(2)=sum(delta.*x)/m;




end


