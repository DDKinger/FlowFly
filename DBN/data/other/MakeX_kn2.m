clc;
clear;
TimeK = 3;  %time intervals
X_input = [];
X_input_temp = [];
n_observation_total = 82;
for obj_num = 1:n_observation_total
    if obj_num==33||obj_num==41||obj_num==47||obj_num==56....
            ||obj_num==3||obj_num==19||obj_num==21||obj_num==35....
            ||obj_num==38||obj_num==43||obj_num==54||obj_num==59....
%            ||obj_num==72       %69 stations, comment if 70 stations
        continue;
    end
    X_input_temp2 = [];
    for week_num = 1:12
        X_input_temp1 = [];
        for j = 1:TimeK
            FlowVeh5Minutes = xlsread(['G:\DateSet\Traffic\PeMs\2016FLOW\' num2str(obj_num) '-' num2str(week_num) '.xls'],'B2:B1443');
            X_input_temp1 = [X_input_temp1 FlowVeh5Minutes(j:j+12*24*5-1)];
        end
        X_input_temp2 = [X_input_temp2; X_input_temp1];
        fprintf(1,' %d-%d\n',obj_num,week_num);
    end
    X_input = [X_input X_input_temp2];
end