%% clear  
clc; clear;

%load toolbox
addpath(genpath('DeepLearnToolbox-master'));

% Load input Data
data = dlmread('Yeast input.txt');

% load output data
ID = fopen('Yeast output.txt');
C = textscan(ID,'%s');
fclose(ID);
% formated 
text = C{1, 1};


trainX = zeros(size(text, 1), 10);
for m = 1:size(text,1)
   if(strcmp(text{m}, 'CYT') == 1)
       trainX(m, 1) = 1;
       
   elseif(strcmp(text{m}, 'NUC') == 1)
       trainX(m, 2) = 1;
       
   elseif(strcmp(text{m}, 'MIT') == 1)
       trainX(m, 3) = 1;
       
   elseif(strcmp(text{m}, 'ME3') == 1)
       trainX(m, 4) = 1;
       
   elseif(strcmp(text{m}, 'ME2') == 1)
       trainX(m, 5) = 1;
       
   elseif(strcmp(text{m}, 'ME1') == 1)
       trainX(m, 6) = 1;
       
   elseif(strcmp(text{m}, 'EXC') == 1)
       trainX(m, 7) = 1;
       
   elseif(strcmp(text{m}, 'VAC') == 1)
       trainX(m, 8) = 1;
       
   elseif(strcmp(text{m}, 'POX') == 1)
       trainX(m, 9) = 1;
       
   else % must be ERL
       trainX(m, 10) = 1;
   end
end

% set a random variable 
random = randperm(size(trainX,1)); 
%randomly split the data sets
trainS = random(1:floor(1484*0.65));
trainI = data(trainS,:);
trainO = trainX(trainS,:);
testS = random(floor(1484*0.65)+1:1484);
testo = trainX(testS,:);
testI = data(testS,:);
%
% setting up the neural network
net = nnsetup([8 3 10]);
%   perceptron works
net.activation_function = 'perceptron';

% learning rate
net.learningRate = .5; 
% net variable 
net.testing = 0;
net.plotting = 1;
net.plotting2 = 1;
%%
% opts variable 
opts.numepochs = 500; 
opts.batchsize = 1;
opts.plot = 0;
% nnt need 4 varbiables 
[net, ~] = nntrain(net, trainI, trainO, opts);
net.testing = 1;
net.plotting = 1;
net.plotting2 = 0;
opts.numepochs = 1;
[net, ~] = nntrain(net, testI, testo, opts);
