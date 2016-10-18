function out = mapFeature(X1, X2)  
%MAPFEATURE Summary of this function goes here
%   Detailed explanation goes here
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end


