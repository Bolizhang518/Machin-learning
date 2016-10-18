%read data

data=load('autompg.txt');

s = randperm(392);
training = data(s(1:280),:);
test = data(s(281:392),:);

mpg=training(:,1);cylinders=training(:,2);
displacement=training(:,3);horsepower=training(:,4);
weight=training(:,5);acceleration=training(:,6);
modelYr=training(:,7);origin=training(:,8);

Y=mpg;
X1=cylinders;X2=displacement;
X3=horsepower;X4=weight;
X5=acceleration;X6=modelYr;
X7=origin;

%TEST SET FROM MPG TO ORIGIN
Z0=test(:,1);Z1=test(:,2);
Z2=test(:,3);Z3=test(:,4);
Z4=test(:,5);Z5=test(:,6);
Z6=test(:,7);Z7=test(:,8);

figure;

%cylinders
xstart=2; xend=12;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X1,x);
%error
train_error(X1,Y,a,b,c,d,e,'cylinders');
test_error(Z1,Y,a,b,c,d,e,'cylinders');

graph0_4(y1,y2,y3,y4,y5,Z1,Z0,x,xstart,xend,'cylinders',1);

%displacement
xstart=40;xend=450;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X2,x);

%error
train_error(X2,Y,a,b,c,d,e,'disp');
test_error(Z2,Y,a,b,c,d,e,'disp');

graph0_4(y1,y2,y3,y4,y5,Z2,Z0,x,xstart,xend,'displacement',2);

%horsepower
xstart=30;xend=210;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X3,x);

%error
train_error(X3,Y,a,b,c,d,e,'horsepower');
test_error(Z3,Y,a,b,c,d,e,'horsepower');

graph0_4(y1,y2,y3,y4,y5,Z3,Z0,x,xstart,xend,'horsepower',3);
%weight
xstart=1440;xend=5500;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X4,x);

%error
train_error(X4,Y,a,b,c,d,e,'weight');
test_error(Z4,Y,a,b,c,d,e,'weight');

graph0_4(y1,y2,y3,y4,y5,Z4,Z0,x,xstart,xend,'weight',4);

%acc
xstart=7;xend=35;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X5,x);

%error
train_error(X5,Y,a,b,c,d,e,'acc');
test_error(Z5,Y,a,b,c,d,e,'acc');

graph0_4(y1,y2,y3,y4,y5,Z5,Z0,x,xstart,xend,'acc',5);

%modelYr
xstart=55;xend=75;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X6,x);

%error
train_error(X6,Y,a,b,c,d,e,'modelYr');
test_error(Z6,Y,a,b,c,d,e,'modelYr');

graph0_4(y1,y2,y3,y4,y5,Z6,Z0,x,xstart,xend,'modelYr',6);

%origin
xstart=0;xend=3.5;
x=xstart:0.1:xend;

[y1,y2,y3,y4,y5,a,b,c,d,e]=regress4(Y,X7,x);

%error
train_error(X7,Y,a,b,c,d,e,'origin');
test_error(Z7,Y,a,b,c,d,e,'disp');

graph0_4(y1,y2,y3,y4,y5,Z7,Z0,x,xstart,xend,'origin',7);
%close;