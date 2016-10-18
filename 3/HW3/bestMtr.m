
%===========================================================
%score matrix multiclass classifier                                          
%score the true y vlalu
%thresholds determine TP, TN, FP, FN
%AUC AUPRC     2-plot the area
%=====================================================================
function [x1, y1, x2, y2, AUC, AUPRC] = matrix(pred, trueyind, threshold)


m = length(threshold);
x1 = zeros(m, 1); y1 = zeros(m ,1);x2 = zeros(m, 1);y2 = zeros(m, 1);

%resize score matrix
pred = (pred - mean(mean(pred))) ./ std(reshape(pred, 1, numel(pred)));
%map thme to range 1,0
map = (pred - min(min(pred))) ./ (max(max(pred)) - min(min(pred)));

for i = 1:m
    score_class = map;
    %if scores bigger than threshold then 1, otherwise 0 
    score_class(score_class <= threshold(i)) = 0;
    score_class(score_class > threshold(i)) = 1;
    %map matrix indice to linear indice
    idx = sub2ind(size(score_class), trueyind', 1:size(score_class, 2));
    %if scores corresponding to true y class is 1,  then 'classified' 
    positiveidx = (score_class(idx) == 1); 
     positivemat = score_class(:, positiveidx); 
    TP = sum(sum(positivemat) == 1); 
    FP = sum(sum(positivemat) > 1);
   %iftrue y class is not 1 
    negativeidx = (score_class(idx) ~= 1);
  
    negativemat = score_class(:, negativeidx); 

    % TN FN 
    TN = sum(sum(negativemat)  ~= 1);  
    FN = sum(sum(negativemat) == 1); 
 
   
    x1(i) = FP/(FP+TN);
    y1(i) = TP/(TP+FN);
    x2(i) = TP/(TP + FN);
    y2(i) = TP/(TP + FP);
end

[x1, I] = sort(x1);
[x2, J] = sort(x2);
 y1 = y1(I);
 y2 = y2(J); 

%take out numbers that are not NaN

idx1 = isnan(y1) | isnan(x1); idx2 = isnan(y2) | isnan(x2);
x1new = x1(~idx1); x2new = x2(~idx2);
y1new = y1(~idx1); y2new = y2(~idx2);
% get width hight for the area 
widthAUC = lagmatrix(x1new, -1) - x1new;
widthAUPRC = lagmatrix(x2new, -1) - x2new;
heightAUC = lagmatrix(y1new, -1) + y1new;
heightAUPRC = lagmatrix(y2new, -1) + y2new;  
AUC = sum(widthAUC(1:(end -1)) .* heightAUC(1:(end -1)) ./ 2);
AUPRC = sum(widthAUPRC(1:(end -1)) .* heightAUPRC(1:(end -1)) ./ 2);

end

