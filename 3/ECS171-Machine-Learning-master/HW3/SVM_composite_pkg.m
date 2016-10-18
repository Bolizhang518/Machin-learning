
%Implement with fitcsvm function
function [testclass, errorrate, varargout] = SVM_composite_pkg(x, y, kernel, showplot)
%The same as multiclass function except we have to combine labels from various reponse variables together at first. 

m = size(y, 2);
y1 = y(:, 1);
for i = 1:(m-1)
   y1 = strcat(y1, '_', y(:, i+1));  %concatenate all labels together with underscore
end
yclass = unique(y1);
n = length(yclass);
yind = cell2mat(cellfun(@(x) find(strcmp(yclass, x)), y1, 'UniformOutput' , false)); %used for ROC and PRC plot only.

predcomb = zeros(n, size(x, 1));

for i = 1:n
    class = yclass{i};
    yval = cellfun(@(x) strcmp(x, class), y1, 'UniformOutput', false);
    yval = double(cell2mat(yval));
    yval(yval == 0) = -1;
    model = fitcsvm(x, yval,  'KernelFunction', kernel);
    cv = crossval(model);
    [~,score] = kfoldPredict(cv);
    predcomb(i, :) = score(:, 2)';
end

[~, I] = max(predcomb);
testclass = yclass(I);
errorrate = mean(~strcmp(testclass, y1));

if showplot
    [X1, Y1, X2, Y2, AUC, AUPRC] = ROCPRCcurve(predcomb,  yind, 0:0.000001:1);
    subplot(1,2,1)
    plot(X1, Y1, '-r', 'LineWidth',2)
    xlabel('False Positive Rate')
    ylabel('True Positive Rate')
    title(sprintf('ROC curve for Composite SVM with %s kernel', kernel))
    hold on
    plot([0, 0.5, 1], [0, 0.5, 1], 'k--')
    hold off
    
    subplot(1,2,2)
    plot(X2, Y2, '-b', 'LineWidth',2)
    xlabel('Recall')
    ylabel('Precision')
    title(sprintf('Precison-Recall Curve for Composite SVM with %s kernel', kernel))
    
     area = struct('AUC', AUC, 'AUPRC', AUPRC);
    varargout{1} = area;  %output a struct with AUC and AUPRC
end

end