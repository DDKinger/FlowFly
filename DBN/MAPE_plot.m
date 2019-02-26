figure(3);
head = 10;
tail = 200;
x0 = head:tail;
mapetrain0 = MAPE_train(head:tail);
mapetest0 = MAPE_test(head:tail);

inteval =10;
x = x0(1:inteval:end);
mapetrain = mapetrain0(1:inteval:end);
mapetest = mapetest0(1:inteval:end);
%plot(x,mapetest,'k.-');
plot(x,mapetrain,'k.-',x,mapetest,'rx-','LineWidth',0.8);
%plot(x,mapetrain,'k-',x,mapetest,'r-','LineWidth',1.2);
set(gca, 'FontSize',15);
xlabel('Epoch');
ylabel('MAPE (%)');
%ylabel('MAE');
legend('MAPE on training set','RMSE on testing set');