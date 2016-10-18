%read data
data=load('autompg.txt');

s = randperm(392); 
training = data(s(1:280),:);
test = data(s(281:392),:);

mpg=training(:,1);
cylinders=training(:,2);
displacement=training(:,3);
horsepower=training(:,4);
weight=training(:,5);
acceleration=training(:,6);
modelYr=training(:,7);
origin=training(:,8);

%training set
X0=mpg;
X1=cylinders;
X2=displacement;
X3=horsepower;
X4=weight;
X5=acceleration;
X6=modelYr;
X7=origin;

%test set
Z0=test(:,1);
Z1=test(:,2);
Z2=test(:,3);
Z3=test(:,4);
Z4=test(:,5);
Z5=test(:,6);
Z6=test(:,7);
Z7=test(:,8);

%Linear regression for training set
w0=LRSp5(X0,X1,X2,X3,X4,X5,X6,X7,0);
w1=LRSp5(X0,X1,X2,X3,X4,X5,X6,X7,1);
w2=LRSp5(X0,X1,X2,X3,X4,X5,X6,X7,2);

%test error
s1=0;s2=0;s3=0;

for i=1:112,
    s1=s1+(Z0(i)-w0)^2;
    
    Zi1=[1;Z1(i);Z2(i);Z3(i);Z4(i);Z5(i);Z6(i);Z7(i)];
    s2=s2+Zi1'*w1;
    
    Zi2=[1;Z1(i);Z1(i).^2;Z2(i);Z2(i).^2;Z3(i);Z3(i).^2;Z4(i);Z4(i).^2;Z5(i);Z5(i).^2;Z6(i);Z6(i).^2;Z7(i);Z7(i).^2]
    s3=s3+Zi2'*w2;
   
end    
s1=s1^(1/2);s2=s2^(1/2);s3=s3^(1/2);

fprintf('test errors for three functions are \n%.2f\n%.2f\n%.2f\n', s1,s2,s3);




