function w = LRSp3(Y,X,n)
    if n==0,
        X=ones(length(X),1);    
    elseif n==1,
        X=[ones(length(X),1),X];
    elseif n>1,
        single=X;
        
        indices=1:(n-1);
        for i = indices,
            X=[X,single.^(i+1)];          
        end
        X=[ones(length(X),1),X];
    end
    w=pinv((X'*X))*X'*Y;
        
        