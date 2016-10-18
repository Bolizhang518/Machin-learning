plotData(x_training, y_training)

p=polyfit(x_training,y_training,1);   %POLYNOMIAL VURVE FITTING
x_training=linspace(min(x_training),max(x_training));  
y_training=polyval(p,x_training);
plot(x_training,y_training,'*',x_training,y_training); 