%training error
function train_err(X,Y,a,b,c,d,e,name)
s1=0;s2=0;s3=0;s4=0;s5=0;
for i=1:280,
    s1=s1+(Y(i)-a)^2;
    s2=s2+(Y(i)-(b(1)+X(i)*b(2)))^2;
    s3=s3+(Y(i)-(c(1)+X(i)*c(2)+X(i)^(2)*c(3)))^2;
    s4=s4+(Y(i)-(d(1)+X(i)*d(2)+X(i)^(2)*d(3)+X(i)^(3)*d(4)))^2;
    s5=s5+(Y(i)-(e(1)+X(i)*e(2)+X(i)^(2)*e(3)+X(i)^(3)*e(4)+X(i)^(4)*e(5)))^2;
end    
s1=s1^(1/2);s2=s2^(1/2);s3=s3^(1/2);s4=s4^(1/2);s5=s5^(1/2);
fprintf(name);
fprintf('-mpg: training errors for five functions are \n%.2f\n%.2f\n%.2f\n%.2f\n%.2f.\n', s1,s2,s3,s4,s5);
