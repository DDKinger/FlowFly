x1 = datenum('00:00');
x2 = datenum('23:59');
x = linspace(x1,x2,24*12*1);
% ob_indexs = [34, 7];       %�۲�����
% for ob = 1:length(ob_indexs)
%     ob_index = ob_indexs(ob);
%     for k = 0:3    %k������9
%         figure();
%         y1 = Y_test(24*12*k+1:24*12*(k+1),ob_index)*norm;
%         y2 = Y_predict(24*12*k+1:24*12*(k+1),ob_index)*norm;
%         plot(x,y1,'k.-',x,y2,'r.-','LineWidth',1);
%         legend('Observed traffic flow','Predicted traffic flow','Location','southeast');
%  %       legend('�۲⳵����','Ԥ�⳵����');
%         datetick('x',15); %time15,date6
%         xlabel('Time');
%         ylabel('Flow  ( Veh / 5 mins )');
%        % str = ['station ' num2str(ob_index)];
%        % title(str);
%     end
% end

%% for weekday data and weekend difference show
% ob_indexs = [28];
% for ob = 1:length(ob_indexs)
%     ob_index = ob_indexs(ob);
%     for k = 0:3    %k������9
%         figure();
%         y = Y_test(24*12*k+1:24*12*(k+1),ob_index);
%         plot(x,y,'k.-','LineWidth',1.2);
%         set(gca, 'FontSize',14);
%         datetick('x',15); %time15,date6
%         xlabel('Time');
%         ylabel('Flow  ( Veh / 5 mins )');
%     end
% end

%%
% figure(2);
% y3 = Y_output(24*12*(k+50)+1:24*12*(k+55),ob_index)*norm;% *norm;
% plot(x,y3,'k.-');
% datetick('x',6);

% set(gca,'XTickLabel',dt);
% legend Y_train Y_test;

%% for zhongwen lunwen
ob_indexs = 7;       %�۲�����
for ob = 1:length(ob_indexs)
    ob_index = ob_indexs(ob);
    for k = 0:1    %k������9
        figure();
        y1 = Y_test(24*12*k+1:24*12*(k+1),ob_index)*norm;
        y2 = Y_predict(24*12*k+1:24*12*(k+1),ob_index)*norm;
        plot(x,y1,'k.-',x,y2,'r.-','LineWidth',1);
        legend('ʵ��ֵ','Ԥ��ֵ','Location','southeast');
        datetick('x',15); %time15,date6
        xlabel('ʱ��');
        ylabel('������  ( Veh / 5 mins )');
       % str = ['station ' num2str(ob_index)];
       % title(str);
    end
end
