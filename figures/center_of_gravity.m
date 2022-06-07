fontsize = 14;

close all;

a = 0;
b = -5;
c = 2.5;
moment = -log(.5)/c;

x=-0:.01:2;
y=a-b*exp(-c*x);

plot(x,y);
hold on;

x_point = find(x<=moment);
x_point = numel(x_point);
area(x(1:x_point),y(1:x_point),'EdgeColor','none','FaceColor','g');

area(x(x_point+1:end),y(x_point+1:end),'EdgeColor','none','FaceColor','r');


%xline(0);


xline(moment,'--r');

plot(moment,a-b*exp(-c*moment),'k.','MarkerSize',25);
text(moment+.025,a-b*exp(-c*moment)," " + moment,'interpreter','latex','fontsize',fontsize);

title("$a - b e^{-ct}, a=" + a + ", b = " + b + ", c = " + c + "$",'interpreter','latex','fontsize',fontsize);
%text(.35,4.5,"$\int^{" + moment + "}_{0} " + -b + "  e^{-" + c + " t} dt = 1.0$",'interpreter','latex','fontsize',fontsize);
annotation('textarrow',[.44,.2],[0.65,0.2],'string',"$\int_{0}^{" + moment + "} " + -b + "  e^{-" + c + " t} dt = 1.0$",'interpreter','latex','fontsize',20);
annotation('textarrow',[.4,.3],[.3,.2],'string',"$\int_{" + moment + "}^{\inf} " + -b + "  e^{-" + c + " t} dt = 1.0$",'interpreter','latex','fontsize',20);

set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');

