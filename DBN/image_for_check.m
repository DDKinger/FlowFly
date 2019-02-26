% x1 = datenum('00:00');
% x2 = datenum('23:59');
x1 = datenum('04-04-2016');
x2 = datenum('04-09-2016');
x = linspace(x1,x2,24*12*5);
for i = 60:60
    figure(i);
    set(gcf,'Position',get(0,'ScreenSize'))
    for k = 0:5:59
        subplot(3,4,k/5+1);
        y = Y_output(24*12*k+1:24*12*(k+5),i);% *norm;
        plot(x,y,'k-+');
        datetick('x',6); %time15,data6
    end
end
set(gcf,'Position',get(0,'ScreenSize'))