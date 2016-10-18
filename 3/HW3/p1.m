%read data
clear all; close all; clc

data = xlsread('ecs171.dataset.xlsx');
genes = importdata('ecs171.genes.txt');
%label variable 

groRate =data(:,6); 
genesName = data(1,7:4497);
setdata = data(1:194,1:5);
subdata = data(:,1:4487);
genesId=genes(:,1);

x = subdata(:,2:4487);
y = subdata(:,1)
% construct the lasso fit using ten-fold cross validation
%incldue the fit infor output so it can plot result

[B FitInfo] = lasso(x,y,'CV',10);
lassoPlot(B, FitInfo, 'plotType', 'CV');


