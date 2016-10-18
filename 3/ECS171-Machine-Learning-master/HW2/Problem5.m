%Problem5

%One hidden layer with 9 nodes is the best model according to 
%error in problem4. 

%We use the retrain the whole dataset for prediction.
newsamp = [zeros(1, 10), 0.50, 0.49, 0.52, 0.20, 0.55, 0.03, 0.50, 0.39];
prob = ANN_pred(permyeast, 5000, 0.1, newsamp);

%Which class it should be in?
[~, category] = max(prob);

 %[0.3955, 0.5450, 0.0097, 0.0217, 0.0000, 0.0000, 0.0000, 0.0127, 0.0000,
 %0.0001]
 %We should classify the sample into the second class which is NUC.
