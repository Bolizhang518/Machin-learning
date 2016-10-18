function p = predict(theta, X)
% Number of training examples
m = size(X, 1);

p = zeros(m, 1);

p = sigmoid(X* theta); 
index_1 = find(p >= 0.5); 
index_0 = find(p < 0.5); 

p(index_1) = ones(size(index_1)); 
p(index_0) = zeros(size(index_0));
end

