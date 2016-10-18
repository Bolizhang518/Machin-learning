%Implement with fitcsvm function
function [testclass, errorrate, varargout] = SVM_multiclass_pkg(x, y, kernel, showplot)
%apply one vs all multiclass SVM and do prediction
%showplot                   show PRC and ROC plots for all kernels.
%varargout                  if showplot is true, AUC and AUPRC are also given

yclass = unique(y);
n = length(yclass);
yind = cell2mat(cellfun(@(x) find(strcmp(yclass, x)), y, 'UniformOutput' , false)); %True class number, used for ROC and PRC plot only.

predcomb = zeros(n, size(x, 1));

for i = 1:n
    class = yclass{i};
    yval = cellfun(@(x) strcmp(x, class), y, 'UniformOutput', false);
    yval = double(cell2mat(yval));
    yval(yval == 0) = -1;   %change y from cell string to double array
    model = fitcsvm(x, yval,  'KernelFunction', kernel);  %fit binary svm model for each class
    cv = crossval(model);
    [~,score] = kfoldPredict(cv);  %perform 10 folds CV and get the score in the binary model
    predcomb(i, :) = score(:, 2)'; %put the scores into a score matrix for output and plot
end

[~, I] = max(predcomb);
testclass = yclass(I);
errorrate = mean(~strcmp(testclass, y)); %get class and error

if showplot
    [X1, Y1, X2, Y2, AUC, AUPRC] = ROCPRCcurve(predcomb,  yind, 0:0.000001:1);
    subplot(1,2,1)
    plot(X1, Y1, '-r', 'LineWidth',2)
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    title(sprintf('ROC curve with %s kernel', kernel))
    hold on
    plot([0, 0.5, 1], [0, 0.5, 1], 'k--')
    hold off
   
    subplot(1,2,2)
    plot(X2, Y2, '-b', 'LineWidth',2)
    xlabel('Recall')
    ylabel('Precision')
    title(sprintf('Precison-Recall Curve with %s kernel', kernel))
    
    area = struct('AUC', AUC, 'AUPRC', AUPRC);
    varargout{1} = area;  %output a struct with AUC and AUPRC
    
end


end

