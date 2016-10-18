load ('fisheriris' )

xdata=meas(51:end,3:4);
group = species(51:end,1);

p=0.5;
[train,test] = crossvalind('HoldOut', group,p);

trainsample = xdata(train,:)
trainlable = group(train,1)
testsample = xdata(train,:)
testlable = group(train,1)
%% 5 fould 
numfolds=5;
indices = crossvalind('kfold',trainlable,numfolds);

sigma = 2.^(-5:1:5);
c  = 2.^(-5:1:5);

[Bestsigma, Bestc] = BestParametersRBF(trainsample,...
    trainlable,sigma,c,indices,numfolds);
cause you said we have 4 different multi-class classifiers . 
so I think the 1st one is binary classifiers, build binary classifiers and then use them to do multi-class classification
create new labels using the "medium" and "stress" columns combined. Then perform svm as before.