clc;
clear;
TimeK = 8;  %time intervals
X_input = [];
X_input_temp = [];
n_observation_total = 82;
row_range = ['B2:B' num2str(1441+TimeK-1)];
fprintf(1,'Row Range %s\n', row_range);
for obj_num = 1:n_observation_total
    if obj_num==33||obj_num==41||obj_num==47||obj_num==56....
            ||obj_num==3||obj_num==19||obj_num==21||obj_num==35....
            ||obj_num==38||obj_num==43||obj_num==54||obj_num==59....
            ||obj_num==72       %69 stations, comment if 70 stations
        continue;
    end
    X_input_temp2 = [];
    for week_num = 1:12
        X_input_temp1 = [];
        for j = 1:TimeK
            FlowVeh5Minutes = xlsread(['/home/dd/DD/dataset/2016FLOW/' num2str(obj_num) '-' num2str(week_num) '.xls'],row_range); %1441+TimeK-1
            X_input_temp1 = [X_input_temp1 FlowVeh5Minutes(j:j+12*24*5-1)];
        end
        X_input_temp2 = [X_input_temp2; X_input_temp1];
        fprintf(1,' %d-%d\n',obj_num,week_num);
    end
    X_input = [X_input X_input_temp2];
end