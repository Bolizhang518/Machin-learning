function tserr = ANN_onelayer(trainMAT, iter, alpha, nodenum, testMAT)
%Build a one layer ANN with 10 output nodes, 3, 6, 9, 12 hidden nodes, and 8 input nodes
%tserr     test error for various nodes
%nodenum   hidden node numbers to test for error 

tserr = zeros(1, length(nodenum));%record test error for different hidden nodes

for n = 1 : length(nodenum)
 node = nodenum(n);
 rng(171);
 w1 = -1 + 2 * rand(9, node);
 rng(2);
 w2 = -1 + 2 * rand((node + 1), 10);

 
 for m = 1:iter
    %use stochastic descent
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
        delta2 = summation(2:end) .* (1 - a2) .* a2;
        grad2 = - [1, trainMAT(i, 11:end)]' * delta2;
        w1 = w1 - alpha *grad2;
        
    end

 end
 tsmat = [ones(size(testMAT, 1), 1), testMAT(:, 11:end)];
 a2mat = 1 ./ (1 +exp(-(tsmat * w1)));
 a2matone = [ones(size(a2mat, 1), 1), a2mat];
 a3mat = 1 ./ (1 +exp(-(a2matone * w2)));%Updating activation function values in each layer
 [~, predclass] = max(a3mat, [], 2);
 [~, trueclass] = max(testMAT(:, 1:10), [], 2);
 err = sum(predclass ~= trueclass) / length(predclass);
 tserr(n) = err;   
end

end

