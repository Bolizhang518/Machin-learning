function [w, X] = LRSp3(Y,X1,X2,X3,X4,X5,X6,X7,n)
    if n==0,
        X=(ones(280,1));
    elseif n==1,
        X=[ones(280,1),X1,X2,X3,X4,X5,X6,X7];
    elseif n==2,
        X=[ones(280,1),X1,X1.^2,X2,X2.^2,X3,X3.^2,X4,X4.^2,X5,X5.^2,X6,X6.^2,X7,X7.^2];
    end
    %X1,X2,X3,X4,X5,X6,X7,X
    w=pinv((X'*X))*X'*Y;    
        
        