function graph0_4(y1,y2,y3,y4,y5,Z,mpg,x,xstart,xend, name,n)

subplot(2,4,n);

mask1=mpg<18.6;
mask2=mpg>=18.6 & mpg < 27;
mask3=mpg>=27;
        
scatter(Z(mask1),mpg(mask1),'blue');
hold on;
scatter(Z(mask2),mpg(mask2),'r');
hold on;
scatter(Z(mask3),mpg(mask3), 'green');
hold on
plot(x,y1,'r');
hold on;
plot(x,y2,'g');
hold on;
plot(x,y3,'blue');
hold on;
plot(x,y4,'black');
hold on;
plot(x,y5,'yellow');

legend('data','0th','1st','2nd','3rd','4th')
xlabel(name);
ylabel('mpg');
axis tight;
axis([xstart xend 0 50]);