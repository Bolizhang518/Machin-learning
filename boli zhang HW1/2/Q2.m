x = [cylinders,displacement,horsepower,weight,accelration,model,origin];
varNames  = {'cylinders';'displacement';'horsepower';'weight';'accelration';'model';'origin'};
figure
gplotmatrix(x,[],mpg,[ 'c' 'b' 'm' 'g' 'y' 'r' 'r'],[],[],false);
%text([.08 .24 .43 .66 .83], repmat(-.1,1,5), varNames, 'FontSize',8);
%text(repmat(-.12,1,5), [.86 .62 .41 .25 .02], varNames, 'FontSize',8, 'Rotation',90);
