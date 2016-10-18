data = load('autompg.txt');
mpg = data(:,1);
cylinders = data(:,2);
displacement = data(:,3);
horsepower = data(:,4);
weight = data(:,5);
accelration = data(:,6);
model  = data(:,7);
origin = data(:,8);

sortmpg = sort(mpg);
length_mpg = length(sortmpg);
threshold1 = (length_mpg)/3;
threshold2 = 2*((length_mpg)/3);

mpgthreshold1 = sortmpg(fix(threshold1),1);
mpgthreshold2 = sortmpg(fix(threshold2),1);
mpgthreshold1
mpgthreshold3