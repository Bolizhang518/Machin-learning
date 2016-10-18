%Problem1:

% Read in the data
[numdata, textdata] = xlsread('ecs171.dataset.xlsx', 'data');

% modify dataset
genename = textdata(1, 7:end); %all gene names.
str_data = textdata(2:end, 1:5); %first five columns all string type data 194*5
num_data = numdata(:, 1:(end - 1)); %other num data including growthrate and 4496 gene expression 194*4487

%For problem1, we only use num_data
y = num_data(:, 1);
x = num_data(:, 2:end);

% Lasso function
[b0, fitinfo0] = lasso(x, y, 'CV', 10);
%lambda1 = fitinfo0.Lambda1SE;
%num_nonzero1 = sum(b0(:, fitinfo0.Index1SE) ~= 0);
%mse1 = fitinfo0.MSE(fitinfo0.Index1SE);


%lassoPlot(b0,fitinfo0,'PlotType','CV');
%lassoPlot(b0,fitinfo0,'PlotType','Lambda','XScale','log');

%My own lasso function
%[b, fitinfo] = lassofit(x, y, 1:0.5:10);
%lambda2 = fitinfo.lambda_val;
%num_nonzero2 = sum(b ~= 0);
%mse2 = fitinfo.minMSE;
