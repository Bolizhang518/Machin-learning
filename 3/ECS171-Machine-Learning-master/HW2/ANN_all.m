function [beta, trerr] = ANN_all(trainMAT, iter, alpha)
%The one hidden layer model for all data, variable meanings are the same. 
%This is for training the whole dataset.

 %Random generate initial W (totally 67)
 rng(171);
 w1 = -1 + 2 * rand(9, 3);
 rng(2);
 w2 = -1 + 2 * rand(4, 10);
 
 for m = 1:iter
 
    for i = 1:size(trainMAT, 1)
        
        a2 = 1 ./ (1 +exp(-([1, trainMAT(i, 11:end)] * w1)));
        a3 = 1 ./ (1 +exp(-([1, a2] * w2)));
        
        %update w2
        delta1 = (trainMAT(i, 1:10) - a3) .*  (1 - a3) .* a3;
        grad1 = - [1, a2]' * delta1; 
        w2 = w2 - alpha * grad1;
        %back propgation 
            
        %update w1
        summation = delta1 * w2';
        delta2 = summation(2:4) .* (1 - a2) .* a2;
        grad2 = - [1, trainMAT(i, 11:end)]' * delta2;
        w1 = w1 - alpha *grad2;
        
    end
        
 end
 beta = struct('weightinhid', w1, 'weighthidout', w2);
 
 trmat = [ones(size(trainMAT, 1), 1), trainMAT(:, 11:end)];
 a2mat = 1 ./ (1 +exp(-(trmat * w1)));
 a2matone = [ones(size(a2mat, 1), 1), a2mat];
 a3mat = 1 ./ (1 +exp(-(a2matone * w2)));%Updating activation function values in each layer
 [~, predclass] = max(a3mat, [], 2);
 [~, trueclass] = max(trainMAT(:, 1:10), [], 2);
 err = sum(predclass ~= trueclass) / length(predclass);
 trerr = err;

end

