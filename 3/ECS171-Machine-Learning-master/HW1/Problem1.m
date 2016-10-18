%Problem1: Classify all samples into three equal-sized bins.

%Read in file:
fid = fopen('D:/Rwd/auto-mpg.dat');
mpgdat = textscan(fid, '%f%f%f%f%f%f%f%f%q', 'TreatAsEmpty', '?');
fclose(fid);

%get out the mpg column and transfroms to array
mpg = mpgdat{:, 1};
%sort mpg data
sortedmpg = sort(mpg);
%get the threshold
ind1 = round(398/3);
ind2 = round(398/3*2);
low2med = sortedmpg(ind1);
med2high = sortedmpg(ind2);