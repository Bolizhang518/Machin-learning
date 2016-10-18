function tserr = ANN_twolayers(trainMAT, iter, alpha, nodenum, testMAT)
%Build a two layers ANN with 10 output nodes, 3, 6, 9, 12 hidden nodes, and
%8 input nodes. Variable meanings are the same. 
tserr = zeros(1, length(nodenum));
 
for n = 1 : length(nodenum)

 node = nodenum(n);
 rng(171);
 w1 = -1 + 2 * rand(9, node);
 rng(2);
 w2 = -1 + 2 * rand((node + 1), node);
 rng(3);
 w3 = -1 + 2 * rand((node + 1), 10);
 

 for m = 1:iter
    %use stochastic descent
    for i = 1:size(trainMAT, 1)
        
        a2 = 1 ./ (1 +exp(-([1, trainMAT(i, 11:end)] * w1)));
        a3 = 1 ./ (1 +exp(-([1, a2] * w2)));
        a4 = 1 ./ (1 +exp(-([1, a3] * w3)));
        
        %update w3
        delta1 = (trainMAT(i, 1:10) - a4) .*  (1 - a4) .* a4;
        grad1 = - [1, a3]' * delta1; 
        w3 = w3 - alpha * grad1;
        %back propgation 
        
        %update w2
        summation1 = delta1 * w3';
        delta2 = summation1(2:end) .* (1 - a3) .* a3;
        grad2 = - [1, a2]' * delta2; 
        w2 = w2 - alpha * grad2;
        %back propgation 
        
        %update w1
        summation2 = delta2 * w2';
        delta3 = summation2(2:end) .* (1 - a2) .* a2;
        grad3 = - [1, trainMAT(i, 11:end)]' * delta3;
        w1 = w1 - alpha *grad3;
        
    end

 end
 tsmat = [ones(size(testMAT, 1), 1), testMAT(:, 11:end)];
 a2mat = 1 ./ (1 +exp(-(tsmat * w1)));
 a2matone = [ones(size(a2mat, 1), 1), a2mat];
 a3mat = 1 ./ (1 +exp(-(a2matone * w2)));
 a3matone = [ones(size(a3mat, 1), 1), a3mat];
 a4mat = 1 ./ (1 +exp(-(a3matone * w3))); %Updating activation function values in each layer
 [~, predclass] = max(a4mat, [], 2);
 [~, trueclass] = max(testMAT(:, 1:10), [], 2);
 err = sum(predclass ~= trueclass) / length(predclass);
 tserr(n) = err;
end

end

