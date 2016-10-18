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


trainx = zeros(size(text, 1), 10);

for m = 1:size(text,1)
    
   if(strcmp(text{m}, 'CYT') == 1)
       trainx(m, 1) = 1;
       
   elseif(strcmp(text{m}, 'NUC') == 1)
       trainx(m, 2) = 1;
       
   elseif(strcmp(text{m}, 'MIT') == 1)
       trainx(m, 3) = 1;
       
   elseif(strcmp(text{m}, 'ME3') == 1)
       trainx(m, 4) = 1;
       
   elseif(strcmp(text{m}, 'ME2') == 1)
       trainx(m, 5) = 1;
       
   elseif(strcmp(text{m}, 'ME1') == 1)
       trainx(m, 6) = 1;
       
   elseif(strcmp(text{m}, 'EXC') == 1)
       trainx(m, 7) = 1;
       
   elseif(strcmp(text{m}, 'VAC') == 1)
       trainx(m, 8) = 1;
       
   elseif(strcmp(text{m}, 'POX') == 1)
       trainx(m, 9) = 1;
   else % must be ERL
       trainx(m, 10) = 1;
   end
end

% set a random variable 
random = randperm(size(trainx,1)); 
%randomly split the data sets
trainSplit = random(1:floor(1484*0.65));
trainI = data(trainSplit,:);
trainO = trainx(trainSplit,:);
testS = random(floor(1484*0.65)+1:1484);
testO = trainx(testS,:);
testI = data(testS,:);

totalerr = zeros(3,4); 
for i = 1:4  
    errs = zeros(100,1); 
    for j = 1:100
        
        net1 = nnsetup([8 3*i 10]);
        
        net1.activation_function = 'perceptron';
        net1.learningRate = .05;
        opts.batchsize = 1;
        opts.numepochs = 1;
        
        opts.plot = 0;
        net1.testing = 0;
        net1.plotting2 = 0;
        net1.plotting = 0;
        
        
        [net1, ~] = nntrain2(net1, trainI, trainO, opts);
        net1.testing = 1;
        net1.plotting = 0;
        [net1, erros] = nntrain2(net1, testI, testO, opts);
        errs(j) = erros;
    end
    totalerr(1,i) = mean(errs);
end

for i = 1:4  
    errs = zeros(100,1); 
    for j = 1:100
        
        net1 = nnsetup([8 3*i 10]);
        
        net1.activation_function = 'perceptron';
        net1.learningRate = .05;
        opts.batchsize = 1;
        opts.numepochs = 1;
        
        opts.plot = 0;
        net1.testing = 0;
        net1.plotting2 = 0;
        net1.plotting = 0;
        
        
        [net1, ~] = nntrain2(net1, trainI, trainO, opts);
        net1.testing = 1;
        net1.plotting = 0;
        [net1, erros] = nntrain2(net1, testI, testO, opts);
        errs(j) = erros;
    end
    totalerr(2,i) = mean(errs);
end
for i = 1:4  
    errs = zeros(100,1); 
    for j = 1:100
        
        net1 = nnsetup([8 3*i 10]);
        
        net1.activation_function = 'perceptron';
        net1.learningRate = .05;
        opts.batchsize = 1;
        opts.numepochs = 1;
        
        opts.plot = 0;
       
        net1.testing = 0;
        net1.plotting2 = 0;
        net1.plotting = 0;
        
        
        [net1, ~] = nntrain2(net1, trainI, trainO, opts);
        net1.testing = 1;
        net1.plotting = 0;
        [net1, erros] = nntrain2(net1, testI, testO, opts);
        errs(j) = erros;
    end
    totalerr(3,i) = mean(errs);
end
