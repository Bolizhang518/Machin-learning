%Problem 4
%formula in the two and three hidden layers model are given on paper.

%One layer:
tserr = ANN_onelayer(trainMat, 5000, 0.1, [3, 6, 9, 12], testMat);
%[0.4624, 0.4374, 0.4200, 0.4297]

%Two layers:
tserr2 = ANN_twolayers(trainMat, 5000, 0.1, [3, 6, 9, 12], testMat);
%[0.4509, 0.4566, 0.4528, 0.4566]

%Three layers:
tserr3 = ANN_threelayers(trainMat, 5000, 0.1, [3, 6, 9, 12], testMat);
%[0.4913, 0.4644, 0.4855, 0.4566]