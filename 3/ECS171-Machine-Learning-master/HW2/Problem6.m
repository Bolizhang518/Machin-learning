%Problem6

%We use the formula given in report to calculate the uncertainty

[~, maxind] = max(prob);
s = zeros(1, 10);
s(maxind) = 1;
uncertain_score = sum(abs(prob - s));
uncertain_rate = uncertain_score/5;
%The uncertainty score is 0.7517, in fact, if we got probability 0.5
%for all classes, the uncertainty score would reach to the largest, which
%is 5. The uncertainty rate is 15.03%.  