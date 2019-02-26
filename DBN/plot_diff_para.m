% h_layer
figure(1)
x1 = [2, 3, 4, 5, 6];
y1 = [7.55, 7.78, 8.02, 8.34, 8.97];
plot(x1, y1, 'bo-', 'LineWidth', 1.1);
set(gca, 'xtick', 2:1:6, 'FontSize',20);
xlabel('\it h\_layer');
ylabel('MAPE (%)');

% h_node
figure(2)
x2 = [64,128,150,192,256,320,512];
y2 = [8.22,7.76,7.73,7.62,7.55,7.56,7.53];
plot(x2, y2, 'bo-', 'LineWidth', 1.1);
set(gca, 'xtick', 64:64:512, 'xlim', [64 512], 'FontSize',20);
xlabel('\it h\_node');
ylabel('MAPE (%)');

%c_level
figure(3)
x3 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7];
y3 = [7.87,7.70,7.58,7.56,7.55,7.60,7.61];
plot(x3, y3, 'bo-', 'LineWidth', 1.1);
set(gca, 'xtick', 0.1:0.1:0.7, 'xlim', [0.1 0.7], 'FontSize',20);
xlabel('\it c\_level');
ylabel('MAPE (%)');

%pre_epoch
figure(4)
x4 = [5,10,15,20,25,30,35];
y4 = [7.89,7.58,7.56,7.55,7.59,7.65,7.66];
plot(x4, y4, 'bo-', 'LineWidth', 1.1);
set(gca, 'xtick', 5:5:35, 'xlim', [5 35], 'FontSize',20);
xlabel('\it pre\_epoch');
ylabel('MAPE (%)');

%d
figure(5)
x5 = [1,2,3,4,5,6];
y5 = [7.76,7.55,7.61,7.75,7.92,8.00];
plot(x5, y5, 'bo-', 'LineWidth', 1.1);
set(gca, 'xtick', 1:1:6, 'xlim', [1 6], 'FontSize',20);
xlabel('\it d');
ylabel('MAPE (%)');