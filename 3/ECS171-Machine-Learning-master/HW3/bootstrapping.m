%Problem2:
%Bootstrap, resampling 100 times, 0.95 CI
function conf = bootstrapping(data, n, varargin)

if nargin == 3
    testdata = varargin{1};
else 
    testdata = data(:, 2:end);
end

pred = zeros(n, size(testdata, 1));
conf = zeros(size(testdata, 1), 2);

for i = 1:n
    resample = datasample(data, size(data, 1), 1);
    x = resample(:, 2:end);
    y = resample(:, 1);
    [beta, fitinfo] = lasso(x, y, 'CV', 10);
    beta = beta(:, fitinfo.Index1SE);
    pred(i,: ) = ((bsxfun(@minus, testdata, mean(x))) * beta + mean(y))';
end

mn = mean(pred);
sd = std(pred);
z = 1.96;
conf (:, 1) = (mn - z * sd)';
conf(:, 2) = (mn + z * sd)';
aa=mean(conf(:,1))
end
