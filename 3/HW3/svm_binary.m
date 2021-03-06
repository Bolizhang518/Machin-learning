%============================================================
%use fitcsvm function for train binary support vector machine classifier
%the apply one vs all multiclass classifier and prediction
%then plot the PRC and ROC
%if showplot then give AUC and AUPRC
%=================================================================
function [testdata, err_rate, value] = svm_binary(x, y, kernel, showplot)


ydata = unique(y); %mini y data
n = length(ydata); 
% when showplot true , then plot AUG & AUPRC
%Indicates that for all inputs cell array
yind = cell2mat(cellfun(@(x) find(strcmp(ydata, x)), y, 'UniformOutput' , false));
%prediction combain 
pred = zeros(n, size(x, 1));

for i = 1:n
    class = ydata{i};
    %Indicates that for all inputs cell array
    yval = cellfun(@(x) strcmp(x, class), y, 'UniformOutput', false);
    yval = double(cell2mat(yval));
     % y string to double array 
    yval(yval == 0) = -1; 
     %fit binary svm,compute the elements of the Gram matrix
    model = fitcsvm(x, yval,  'KernelFunction', kernel); 
    cv = crossval(model);
    %get the 10 folds, and scored in binary model
    [~,score] = kfoldPredict(cv); 
    %put the score to matrix and then plot.
    pred(i, :) = score(:, 2)'; 
    
end

[~, I] = max(pred);
testdata = ydata(I);
%get error rate
err_rate = mean(~strcmp(testdata, y)); 
% plot if true 
if showplot
    [x1, ys, x2, yst, AUC, AUPRC] = bestMtr(pred,  yind, 0:0.000001:1);
   
    plot(x1, ys, '-r', 'LineWidth',2)
    xlabel('Fail Rate')
    ylabel('True Rate')
    title(sprintf('ROC kernel', kernel))
    hold on
    plot([0, 0.5, 1], [0, 0.5, 1], 'k--')
    hold on
  
    
    plot(x2, yst, '-b', 'LineWidth',2)
    xlabel('Recall')
    ylabel('Precision')
    title(sprintf('PR Curve kernel', kernel))
    hold on
    plot([0, 0.5, 1], [0, 0.5, 1], 'k--')
    hold off
    
    area = struct('AUC', AUC, 'AUPRC', AUPRC);
    area
    value{1} = area;  %output a struct with AUC and AUPRC
    value{1}
end


end

