%Problem6: Modified solver for multinomial logistic regression(1st order)

function [beta, varargout] = stoc_grad_desc_logi(trainMAT, beta0, alpha, varargin)
%trainMAT   Matrix with frist column as response variable and second column as predictor. 
%beta0      The initial beta value(16 numbers)
%alpha      Learning rate
%beta       The regression coefficients, should be row-wise aligned.
    
    trainMat = trainMAT;
    if size(trainMat, 2) ~= 8 
        disp('WRONG size of data matrix')
    elseif size(trainMat, 1) == 1 %Not enough samples 
        disp('Not enough samples to perform regression')
    else
        %create three variables to represents low, med, high categories for
        %each observation
        Y1 = double(trainMat(:, 1) <= 19);
        Y2 = double(trainMat(:, 1) > 19 &trainMat(:, 1) <= 26.8);
        Y3 = double(trainMat(:, 1) > 26.8); 
        
        %standarization of each variable for logistic regression
        avg = repmat(mean(trainMat), size(trainMat, 1), 1);
        sd = repmat(std(trainMat), size(trainMat, 1), 1);
        trainMat = (trainMat - avg)./sd;
        trainMat = [Y1, Y2, Y3, ones(size(trainMat, 1), 1), trainMat(:, 2:end)];
        
        %Permutation of dataset
        trainmat = trainMat(randperm(size(trainMat, 1)), :);
        betatmp = beta0;
        
        for j = 1:5000
            for i = 1:size(trainmat, 1)
                grad12 =  trainmat(i, 4:end) * trainmat(i, 1) - exp(betatmp(1:8) * trainmat(i, 4:end)')/...
                         (1 + exp(betatmp(1:8) * trainmat(i, 4:end)') + exp(betatmp(9:16) * trainmat(i, 4:end)'))...
                         * trainmat(i, 4:end); %low class gradients
                grad13 = trainmat(i, 4:end) * trainmat(i, 2) - exp(betatmp(9:16) * trainmat(i, 4:end)')/...
                         (1 + exp(betatmp(1:8) * trainmat(i, 4:end)') + exp(betatmp(9:16) * trainmat(i, 4:end)'))...
                         * trainmat(i, 4:end); %median class gradients
                
                grad = [grad12, grad13];
                betanew = betatmp + alpha * grad; %update beta
                betatmp = betanew;
            end
        end 
        stbeta = betatmp; %standarized beta
        %transfer standard beta back to the original beta.
        beta = zeros(1, 16);
        beta(1) = stbeta(1) - sum(stbeta(2:8) .* avg(1, 2:8) ./ sd(1, 2:8));
        beta(2:8) = stbeta(2:8) ./ sd(1, 2:8);
        beta(9) = stbeta(9) - sum(stbeta(10:16) .* avg(1, 2:8) ./ sd(1, 2:8));
        beta(10:16) = stbeta(10:16) ./ sd(1, 2:8);
    end
    
    if nargin == 4 %output MSE
        %Since in logistic regression, we can only get discrete
        %result (class label), the MSE would be misclassification rate
        traintrueclass = findtrueclass(trainMAT(:, 1));
        testtrueclass = findtrueclass(varargin{1}(:, 1));
        trainpredclass = predtrueclass(beta, trainMAT);
        testpredclass = predtrueclass(beta, varargin{1});
        trainmse = mean(traintrueclass ~= trainpredclass);
        testmse = mean(testtrueclass ~= testpredclass);
        varargout{1} = [trainmse, testmse]; %find true class, pred class and ouput to varargout
    end
end

function trueclass = findtrueclass(array)
    class = ones(length(array), 1);
    class(array > 19 & array <= 26.8) = 2;
    class(array > 26.8) = 3;
    trueclass = class;
end

function trueclass = predtrueclass(beta, mat)
    mat(:, 1) = ones(size(mat, 1), 1);
    %exp1, exp2, p1-p3 are implemeted according to properties of
    %multinomial logistic regression.
    exp1 = exp(mat * beta(1:8)');
    exp2 = exp(mat * beta(9:16)');
    p1 = exp1 ./ (ones(size(mat, 1), 1) + exp2 + exp1);
    p2 = exp2 ./ (ones(size(mat, 1), 1) + exp2 + exp1);
    p3 = 1 ./ (ones(size(mat, 1), 1) + exp2 + exp1);
    p = [p1, p2, p3];
    [~, trueclass] = max(p, [], 2);
    
end