function [x1, y1, x2, y2, AUC, AUPRC] = ROCPRCcurve(predcomb, trueyind, threshold)
%predcomb                         score matrix of SVM multiclass model(row--classes, column--194 samples)                                         
%trueyind                            true y class number
%threshold                           thresholds in score matrix to determine TP, TN, FP, FN
%x1, y1, x2, y2                    ROC and PRC curves coordinates
%AUC AUPRC                   area under ROC and PRC

m = length(threshold);
x1 = zeros(m, 1); y1 = zeros(m ,1);x2 = zeros(m, 1);y2 = zeros(m, 1);

%First to standardize score matrix due to high dispersion of scores.
%Next, we should map all standardized scores to range [0,1]
predcomb = (predcomb - mean(mean(predcomb))) ./ std(reshape(predcomb, 1, numel(predcomb)));
scoremat = (predcomb - min(min(predcomb))) ./ (max(max(predcomb)) - min(min(predcomb)));

for i = 1:m
    scoreclass = scoremat;
    scoreclass(scoreclass > threshold(i)) = 1;
    scoreclass(scoreclass <= threshold(i)) = 0;  %scores bigger than threshold classified as 1, otherwise 0 
    idx = sub2ind(size(scoreclass), trueyind', 1:size(scoreclass, 2)); %map matrix indice to linear indice for all scores corresponding to true y class.
    positiveidx = (scoreclass(idx) == 1);  %if scores corresponding to true y class is 1,  then 'classified' 
    positivemat = scoreclass(:, positiveidx);  %get out the classified part
    TP = sum(sum(positivemat) == 1);  %if only 1 class to be classified in, then 'True' &'Classified'
    FP = sum(sum(positivemat) > 1);   %if more than 1 class to be classified in, then, 'False' & 'Classified'
    negativeidx = (scoreclass(idx) ~= 1);%if scores corresponding to true y class is NOT 1,  then 'NOT classified' 
    negativemat = scoreclass(:, negativeidx); %get out the NOT classified part
    TN = sum(sum(negativemat)  ~= 1);  %if more than 1 class or 0 class to be classified in, then, 'False' & 'Not Classified'
    FN = sum(sum(negativemat) == 1); %if only 1 class to be classified in, then 'True' &'Not Classified'
    x1(i) = FP/(FP+TN);
    y1(i) = TP/(TP+FN);
    x2(i) = TP/(TP + FN);
    y2(i) = TP/(TP + FP);
end
[x1, I] = sort(x1); [x2, J] = sort(x2);
y1 = y1(I); y2 = y2(J);  %Sort x and y to plot

%take out the x and y numbers that are not NaN
idx1 = isnan(y1) | isnan(x1); idx2 = isnan(y2) | isnan(x2);
x1new = x1(~idx1); x2new = x2(~idx2);
y1new = y1(~idx1); y2new = y2(~idx2);

%We use small trapezoid areas to approximate AUC and AUPRC
widthAUC = lagmatrix(x1new, -1) - x1new;
widthAUPRC = lagmatrix(x2new, -1) - x2new;
heightAUC = lagmatrix(y1new, -1) + y1new;
heightAUPRC = lagmatrix(y2new, -1) + y2new;   %get width and height for small trapezoids
AUC = sum(widthAUC(1:(end -1)) .* heightAUC(1:(end -1)) ./ 2);
AUPRC = sum(widthAUPRC(1:(end -1)) .* heightAUPRC(1:(end -1)) ./ 2);
end

