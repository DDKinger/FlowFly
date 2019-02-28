TimeK = 1;  %time intervals
Y_output = [];
Y_output_temp = [];
n_observation_total = 82;
for obj_num = 1:n_observation_total
    if obj_num==33||obj_num==41||obj_num==47||obj_num==56....
            ||obj_num==3||obj_num==19||obj_num==21||obj_num==35....
            ||obj_num==38||obj_num==43||obj_num==54||obj_num==59....
        continue;
    end
    Y_output_temp = [];
    for week_num = 1:12
        FlowVeh5Minutes = xlsread(['G:\DateSet\Traffic\PeMs\2016FLOW\' num2str(obj_num) '-' num2str(week_num) '.xls'],'B3:B1442');
        Y_output_temp = [Y_output_temp;FlowVeh5Minutes];
    end
    Y_output = [Y_output Y_output_temp];
    fprintf(1,' %d',obj_num);
end

%save Y_output_70_k2 Y_output