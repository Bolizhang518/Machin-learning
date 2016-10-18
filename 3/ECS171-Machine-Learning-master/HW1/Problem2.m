%Problem2: Create 2d scatterplot.

%Read in file:
fid = fopen('D:/Rwd/auto-mpg.dat');
mpgdat = textscan(fid, '%f%f%f%f%f%f%f%f%q', 'TreatAsEmpty', '?');
fclose(fid);

%get out all columns except the car name and trasfrom into a matrix
mpgmat = cell2mat(mpgdat(1:8));

%we use gplotmatrix to draw the plot
%labels:
label = char('mpg', 'cylinders', 'displacement','horsepower',...
          'weight', 'acceleration', 'model year', 'origin');
%groups
low2med = 19;
med2high = 26.8;
group = ones(length(mpgdat{:, 1}), 1);
group(mpgmat(:, 1) > low2med & mpgmat(:, 1) < med2high) = 2;
group(mpgmat(:, 1) > med2high) = 3;

gplotmatrix(mpgmat, [], group, 'brg', [], [5, 5, 5], 'off', 'hist',...
            label, []);