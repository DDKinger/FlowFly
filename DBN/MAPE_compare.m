figure(2);
head = 10;
tail = 500;
x0 = head:tail;
mapetrain0 = DDBN_mape_train(head:tail);
mapetest0 = DDBN_mape_test(head:tail);
mapetrain1 = CDBN_mape_train(head:tail);
mapetest1 = CDBN_mape_test(head:tail);
mapetrain2 = MLP_mape_train(head:tail);
mapetest2 = MLP_mape_test(head:tail);
mapetrain3 = SDAE_mape_train(head:tail);
mapetest3 = SDAE_mape_test(head:tail);
mapetrain4 = SDBN_mape_train(head:tail);
mapetest4 = SDBN_mape_test(head:tail);

inteval =35;
x = [x0(1:inteval:end),tail];
mapetrain_0 = [mapetrain0(1:inteval:end),mapetrain0(end)];
mapetest_0 = [mapetest0(1:inteval:end),mapetest0(end)];
mapetrain_1 = [mapetrain1(1:inteval:end),mapetrain1(end)];
mapetest_1 = [mapetest1(1:inteval:end),mapetest1(end)];
mapetrain_2 = [mapetrain2(1:inteval:end),mapetrain2(end)];
mapetest_2 = [mapetest2(1:inteval:end),mapetest2(end)];
mapetrain_3 = [mapetrain3(1:inteval:end),mapetrain3(end)];
mapetest_3 = [mapetest3(1:inteval:end),mapetest3(end)];
mapetrain_4 = [mapetrain4(1:inteval:end),mapetrain4(end)];
mapetest_4 = [mapetest4(1:inteval:end),mapetest4(end)];
%plot(x,mapetest,'k.-');
plot(x,mapetest_0,'kd-',x,mapetest_1,'rd-',x,mapetest_2,'gd-',x,mapetest_3,'md-',x,mapetest_4,'bd-','LineWidth',1.2);
% plot(x,mapetrain_0,'k.-',x,mapetest_0,'kd-',x,mapetrain_1,'r.-',x,mapetest_1,'rd-',...
%     x,mapetrain_2,'g.-',x,mapetest_2,'gd-',x,mapetrain_3,'m.-',x,mapetest_3,'md-',...
%     x,mapetrain_4,'b.-',x,mapetest_4,'bd-','LineWidth',1.0);
set(gca, 'FontSize',14);
xlabel('Epoch');
ylabel('MAPE (%)');
%ylabel('RMSE');
% le = legend( 'D-DBN (train)','D-DBN (test)',...
%         'Classical DBN (train)','Classical DBN (test)',...
%         'MLP (train)','MLP (test)',...
%         'SDAE (train)','SDAE (test)',...
%         'Similar DBN (train)','Similar DBN (test)'...
%         );
le = legend('D-DBN','Classical DBN','MLP', 'SDAE','Similar DBN');
legend('boxoff');
set(le,'FontSize',10);