%load G:/MachineLearning/Traffic_Flow_Prediction2/data/X_input_69_k1
obj_num = 16;
week_num = 1;
X_input = xlsread(['D:\dataset\Traffic\PeMs\2016FLOW\' num2str(obj_num) '-' num2str(week_num) '.xls'],'B2:B2017');
%B2:B1441���������ݣ�B1442:B2017��ĩ����
x1 = datenum('00:00');
x2 = datenum('23:59');
x = linspace(x1,x2,24*12*1);
for k = 0:6    %k������6
    figure();
    y1 = X_input(24*12*k+1:24*12*(k+1));
    plot(x,y1,'k.-','LineWidth',1);
    datetick('x',15); %time15,date6
    xlabel('Time');
    ylabel('Flow  ( Veh / 5 mins )');
%    str = ['station ' num2str(ob_index)];
%    title(str);
end