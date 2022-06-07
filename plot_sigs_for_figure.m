fontsize = 18;

figure;
plot (x,signal,'LineWidth',2);
hold on;
%plot(x,myzoh(x,x_sub,double(signal_sub)),'LineWidth',3);
legend({"$V(t)$"},'interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
xlim([11.22 11.27]);
ylim([-.15 .15]);
xlabel('time (s)','interpreter','latex');
ylabel('acceleration','interpreter','latex');

figure;
%plot (x,signal,'LineWidth',2);
plot(x,myzoh(x,x_sub,double(signal_sub)),'LineWidth',2);
hold on;
%plot(x,myzoh(x,x_sub,double(signal_sub)),'LineWidth',2);
%legend({"$V(t)$","$V_{subsample}(t)$"},'interpreter','latex');
legend({"$V_{subsample}(t)$"},'interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
xlim([11.22 11.27]);
ylim([-.15 .15]);
xlabel('time (s)','interpreter','latex');
ylabel('acceleration','interpreter','latex');
