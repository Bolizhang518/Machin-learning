%Problem5 Modified function for 2nd order regression of all features(15 terms).

function [beta, varargout] = polyremod(trainMat, varargin)
%trainMat   Matrix with frist column as response variable and second column as predictor. 
%varargin   represents potential test dataset
%beta       The regression coffecients
%varargout  Only given if testdata is also given, contains MSE and prediction info.
    
    if size(trainMat, 2) ~= 8 
        disp('WRONG size of data matrix')
    elseif size(trainMat, 1) == 1 %Not enough samples 
        disp('Not enough samples to perform regression')
    else
        Xmat = ones(size(trainMat, 1), 15); %The first column is all 1s
        %fill the remaining 14 columns
        for i = 1 : 14
            if mod(i, 2) == 1
                j = 1;
            else j = 2; %identify the polynomial order for each column
            end
            %round(i/2) + 1 would give us the correct column to get the data
            Xmat(:, i+1) = trainMat(:, round(i/2) + 1) .^ j;
        end
        beta = (Xmat'*Xmat)\(Xmat'*trainMat(:, 1)); %beta = (X'X)^-1(X'Y)
               
    end
    
    if nargin == 2  %output MSE and prediction only if test data is given 
        trainMSE = ((trainMat(:, 1) - Xmat * beta)' * (trainMat(:, 1) - Xmat * beta))...
                   /size(trainMat, 1);
        Xmat1 = ones(size(varargin{1}, 1), 15);
        for i = 1:14
            if mod(i, 2) == 1
                j = 1;
            else j = 2;
            end
            Xmat1(:, i+1) = varargin{1}(:, round(i/2) + 1) .^ j;
        end
        pred = Xmat1 * beta; %Yhat = X*Beta
        predMSE = ((varargin{1}(:, 1) - pred)' * (varargin{1}(:, 1) - pred))...
                  /size(varargin{1}, 1);
        varargout{1} = [trainMSE, predMSE]'; %Output MSE which has training and test MSE 
        varargout{2} = pred; %output prediction result.
    end
end