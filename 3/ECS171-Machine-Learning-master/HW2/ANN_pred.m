function prob = ANN_pred(trainMAT, iter, alpha, testMAT)
%Build a one layer ANN with 10 output nodes, 9 hidden nodes, and 8 input nodes
%For testing problem 5 sample only. We choose 9 hidden nodes because of the
%lowest error rate given in problem4.
%prob     probability to be categorised into each class for the sample 

 rng(171);
 w1 = -1 + 2 * rand(9, 9);
 rng(2);
 w2 = -1 + 2 * rand(10, 10);

 
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
 a3mat = 1 ./ (1 +exp(-(a2matone * w2))); %Updating activation function values in each layer

 prob = a3mat;

end

