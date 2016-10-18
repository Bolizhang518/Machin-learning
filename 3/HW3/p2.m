
%% Q2
 
LSL = 0.2957; USL =0.3560 ;                     % Process specifications
capable = @(x)(USL-LSL)./(6* std(x));  % Process capability
ci = bootci(100,capable,groRate)            % BCa confidence interval
%%
%max(groRate)   %1.1826
%min(groRate)   %0
%median(groRate) %0.0603
