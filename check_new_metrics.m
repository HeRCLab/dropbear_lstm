% read Puga's signal
data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
x = data.Segment1.data(:,1);
signal = data.Segment1.data(:,4)';
sample_rate = numel(data.Segment1.data(:,1))/data.Segment1.data(end,1);

% % signal synthesis
% sample_rate = 5000;
% time = 10;
% amps=[1, 2, 3, 0.7, 2.3, 1];
% freqs=[2, 3.7, 7.80, 0.12, 0.54, 1.3];
% phases=[0, 1, 2, 3, 5, 1];
% % synthesize signal
% [x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

% sweep sample period linearly from native down to 0.1 in 100 steps
mi = [];
subsample = [];
for downsample = 1 : -.01 : .01
    tau = 1/(sample_rate*downsample);
    mi_val = mutual_information(x,signal,tau);
    mi = [mi,mi_val];
    subsample = [subsample,downsample];
end

sample_rates = subsample * sample_rate;

plot(sample_rates,mi);
title('mutual information vs sample rate');
xlabel('sample rate');
ylabel('mi');

% sweep history length from 2 to 1000
fnn_gradient1=[];
fnn_gradient2=[];
v_s = var(signal);

for history_length=2:150
    sum_metric1=0;
    sum_metric2=0;
    cnt = numel(signal)-history_length-2;
    for i=1:cnt
        d1 = [signal(i:i+history_length-1);signal(i+1:i+history_length)];
        d2 = [signal(i:i+history_length);signal(i+1:i+history_length+1)];
        [metric1,metric2]=false_nearest_neighbor(d1,d2,v_s);
        sum_metric1 = sum_metric1 + metric1;
        sum_metric2 = sum_metric2 + metric2;
    end
    fnn_gradient1=[fnn_gradient1,sum_metric1/cnt];
    fnn_gradient2=[fnn_gradient2,sum_metric2/cnt];
end

figure;
subplot(2,1,1);
plot(fnn_gradient1);
xlabel('history\_length');
ylabel('FNN metric 1');
subplot(2,1,2);
plot(fnn_gradient2);
xlabel('history\_length');
ylabel('FNN metric 2');
