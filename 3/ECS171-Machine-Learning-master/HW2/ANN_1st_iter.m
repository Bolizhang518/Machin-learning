function beta = ANN_1st_iter(trainMAT, alpha)
%Build a ANN with 10 output nodes, 3 hidden nodes, and 8 input nodes
%only first obsveration will be updated in order to compare with hand calculation.

 %Random generate initial W (totally 67)
 rng(171);
 w1 = -1 + 2 * rand(9, 3);
 rng(2);
 w2 = -1 + 2 * rand(4, 10);

 a2 = 1 ./ (1 +exp(-([1, trainMAT(1, 11:end)] * w1)));
 a3 = 1 ./ (1 +exp(-([1, a2] * w2)));
        
 %update w2
 delta1 = (trainMAT(1, 1:10) - a3) .*  (1 - a3) .* a3;
 grad1 = - [1, a2]' * delta1; 
 w2 = w2 - alpha * grad1;
 %back propgation 
            
 %update w1
 summation = delta1 * w2';
 delta2 = summation(2:4) .* (1 - a2) .* a2;
 grad2 = - [1, trainMAT(1, 11:end)]' * delta2;
 w1 = w1 - alpha *grad2;
        
        
 beta = struct('weightinhid', w1, 'weighthidout', w2);
end

