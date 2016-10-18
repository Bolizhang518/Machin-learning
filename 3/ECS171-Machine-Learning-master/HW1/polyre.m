%Problem3: This is a linear regression solver with an
%accomodation for polynomial basis on a single variable.

function [beta, varargout] = polyre(trainMat, n, varargin)
%We use     varargout to accomodate various order of polynomial regression
%trainMat   Matrix with frist column as response variable and second column as predictor. 
%n          The order of polynomial basis
%varargin   represents potential test dataset
%beta       The regression coffecients
%varargout  Only given if testdata is also given, contains MSE and prediction info.
    
    if size(trainMat, 2) ~= 2 
        disp('WRONG size of data matrix')
    elseif size(trainMat, 1) == 1 %Not enough samples 
        disp('Not enough samples to perform regression')
    else
        Xmat = ones(size(trainMat, 1), n + 1); %The first column is all 1s
        %if n is 0, only constant term will be used for regression. Otherwise, we should fill Xmat with x data.
        if n ~= 0 
            for i = 2 : (n + 1)
                Xmat(:, i) = trainMat(:, 2) .^ (i - 1);
            end
        end
        beta = (Xmat'*Xmat)\(Xmat'*trainMat(:, 1)); %beta = (X'X)^-1(X'Y)
               
    end
    
    if nargin == 3  %output MSE and prediction only if test data is given 
        trainMSE = ((trainMat(:, 1) - Xmat * beta)' * (trainMat(:, 1) - Xmat * beta))...
                   /size(trainMat, 1);
        Xmat1 = ones(size(varargin{1}, 1), n + 1);
        if n~= 0
            for i = 2 : (n + 1)
                Xmat1(:, i) = varargin{1}(:, 2) .^ (i - 1);
            end
        end
        pred = Xmat1 * beta; %Yhat = X*Beta
        predMSE = ((varargin{1}(:, 1) - pred)' * (varargin{1}(:, 1) - pred))...
                  /size(varargin{1}, 1);
        varargout{1} = [trainMSE, predMSE]'; %Output MSE which has training and test MSE 
        varargout{2} = pred; %output prediction result.
    end
end

