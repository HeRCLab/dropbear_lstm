fontsize = 14;

rs = [2500:2500:20000];
ss_snr = [26.9 37.7 42.9 46.8 48.9 50.7 52.1 54.1];
m1_snr = [24.4 32.9 39.4 42.1 46.8 48.9 50.1 46.6];
h1 = [200 700 1100 1300 2200 2200 2200 3200];

m2_snr = [18.9 24.8 30.6 32.4 34.9 37.1 37.5 38.5];
h2 = [100 200 300 400 500 600 700 800];
%%
figure;
hold on;

plot (rs,ss_snr,'b+-');
h = plot (rs,m1_snr,'ro-');

for i=1:size(h1,2)
    if i==size(h1,2)
        text(rs(i)-2000,m1_snr(i)-2,sprintf("$h = %d (%0.0f ms)$",h1(i),h1(i)/rs(i)*1000),'interpreter','latex','fontsize',fontsize);
    else
        text(rs(i)+400,m1_snr(i),sprintf("$h = %d (%0.0f ms)$",h1(i),h1(i)/rs(i)*1000),'interpreter','latex','fontsize',fontsize);
    end
end

legend({'Subsample SNR','Subsample+Model SNR'},'interpreter','latex');
title('Model accuracy','interpreter','latex');
xlabel('Subsample rate ($r_s$)','interpreter','latex');
ylabel('$SNR_{db}$','interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ax = ancestor(h, 'axes');
ax.XAxis.Exponent = 0;

%%
figure;
hold on;
plot (rs,ss_snr,'b+-');
h = plot (rs,m2_snr,'ro-');

for i=1:size(h2,2)
    if i==size(h2,2)-2
        text(rs(i),m2_snr(i)+1,sprintf("$h = %d (%0.0f ms)$",h2(i),h2(i)/rs(i)*1000),'interpreter','latex','fontsize',fontsize);
    elseif i==size(h2,2)-1
        text(rs(i),m2_snr(i)+1,sprintf("$h = %d (%0.0f ms)$",h2(i),h2(i)/rs(i)*1000),'interpreter','latex','fontsize',fontsize);
    elseif i==size(h2,2)
        text(rs(i)-1000,m2_snr(i)+1,sprintf("$h = %d (%0.0f ms)$",h2(i),h2(i)/rs(i)*1000),'interpreter','latex','fontsize',fontsize);
    else
        text(rs(i)+400,m2_snr(i),sprintf("$h = %d (%0.0f ms)$",h2(i),h2(i)/rs(i)*1000),'interpreter','latex','fontsize',fontsize);
    end
end

legend({'Subsample SNR','Subsample+Model SNR'},'interpreter','latex');
title('Model accuracy','interpreter','latex');
xlabel('Subsample rate ($r_s$)','interpreter','latex');
ylabel('$SNR_{db}$','interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ax = ancestor(h, 'axes');
ax.XAxis.Exponent = 0;


%%
% re-training parameters
figure;
hold on;
rs = [2500:2500:20000];
h = plot (rs,b_vals+a_vals,'ro-');
plot (rs,a_vals,'b+-');

legend({'start error','end error'},'interpreter','latex');
title('Retraining parameters, $h/r_s$ = 70 ms','interpreter','latex');
xlabel('Subsample rate ($r_s$)','interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ylim([0 0.03]);
ax = ancestor(h, 'axes');
ax.XAxis.Exponent = 0;


%%
% throughput requirement
figure;
hold on;
rs = [2500:2500:750000];
wcet = 1./rs;
h = rs * 40e-3;
s = 50;
ops = 2*(2*h*s+s+2*s+1)+3*h*s+3*s+2;
%p = plot (rs,ops./wcet/1e9,'b+-');
p = plot (rs,ops./wcet/1e9,'b-');
plot(rs,8*64*4*2*ones(1,numel(rs)),'r-');
plot(rs,6840*.4*2*ones(1,numel(rs)),'g-');

%legend({'minimum ops/s'},'interpreter','latex');
title('Performance requirement, $s$=50, $h/r_s$ = 40 ms','interpreter','latex');
xlabel('Subsample rate ($r_s$)','interpreter','latex');
ylabel('Gops/s','interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ax = ancestor(p, 'axes');
ax.XAxis.Exponent = 0;


%%
% bandwidth requirement
figure;
hold on;
rs = [2500:2500:1200000];
wcet = 1./rs;
h = rs * 40e-3;
s = 50;
ops = 2*(2*h*s+s+2*s+1)+3*h*s+3*s+2;
bytes = 3*(h*s + s);
%p = plot (rs,bytes./wcet/1e9,'b+-');
p = plot (rs,bytes./wcet/2^30,'b-');
plot (rs,8*4e9*64/2^30*ones(1,numel(rs)),'r-');
plot (rs,4320*4*400e6/2^30*ones(1,numel(rs)),'g-');

%legend({'minimum ops/s'},'interpreter','latex');
title('Bandwidth requirement (8-bit wordsize), $s$=50, $h/r_s$ = 40 ms','interpreter','latex');
xlabel('Subsample rate ($r_s$)','interpreter','latex');
ylabel('GB/s','interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ax = ancestor(p, 'axes');
ax.XAxis.Exponent = 0;

%%
fontsize = 14;
markersize = 12;

rs = [2500:2500:20000];
bram = [4 16 23 23 23 23 23 23];
bram = [2 4 6 9 11 13 15 18];

dsp = [3 10 14 14 14 14 14 14];
dsp = [2 3 5 6 7 8 9 11];

lut = [8 24 39 49 70 70 70 95];
lut = [3 6 13 17 19 22 24 27];

latency = [1.23 3.37 5.33 6.16 10.15 10.15 10.15 15.15];
latency = [.57 .78 .99 1.2 1.41 1.62 1.83 3.78];

latency_constraint = 1./rs*1e6;

figure;
hold on;
p = plot(rs,bram,'r-o','LineWidth',2,'MarkerSize',markersize);
plot(rs,dsp,'g-+','LineWidth',2,'MarkerSize',markersize);
plot(rs,lut,'b-*','LineWidth',2,'MarkerSize',markersize);
xlabel('$r_s$','interpreter','latex');
ylabel('%','interpreter','latex');
%title("Parameter Set 1, $80 ms \le h/r_s \le 176 ms$",'interpreter','latex','FontSize',fontsize);
title("Parameter Set 2, $h/r_s = 40 ms$",'interpreter','latex','FontSize',fontsize);

legend({"BRAM utilization","DSP utilization","LUT utilization"},'interpreter','latex');

set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ax = ancestor(p, 'axes');
ax.XAxis.Exponent = 0;

hold off;
figure;
hold on;

p = plot(rs,latency,'c-x','LineWidth',2,'MarkerSize',markersize);
plot(rs,latency_constraint,'m-s','LineWidth',2,'MarkerSize',markersize);
xlabel('$r_s$','interpreter','latex');
ylabel('$\mu s$','interpreter','latex');
%title("Parameter Set 1, $80 ms \le h/r_s \le 176 ms$",'interpreter','latex','FontSize',fontsize);
title("Parameter Set 2, $h/r_s = 40 ms$",'interpreter','latex','FontSize',fontsize);

legend({"latency","latency constraint"},'interpreter','latex');

set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
ax = ancestor(p, 'axes');
ax.XAxis.Exponent = 0;
