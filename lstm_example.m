droptowerdata = importfile_droptower("drop_tower_data.csv");
x = droptowerdata.time';
signal = droptowerdata.test1';

% set up the time axis
time_span = x(end) - x(1);
sample_rate = numel(x)/time_span;
time_offset = x(1);

layers = [sequenceInputLayer(1) lstmLayer(50) lstmLayer(50) fullyConnectedLayer(1) regressionLayer];
layers = [sequenceInputLayer(1) lstmLayer(400) fullyConnectedLayer(1) regressionLayer];

opts = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.05, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1);

train_x = signal(1,1:end-19);
train_y = signal(1,20:end);

net = trainNetwork(train_x,train_y,layers,opts);

%[net_updated,signal_pred] = mypredictAndUpdateState2(net,train_x);
[net_updated,signal_pred] = mypredictAndUpdateState2(net,train_x);

error_power = rms(signal_pred - signal(1,20:end));
signal_power = rms(signal);
snr = log10(signal_power/error_power)*20

close all;
plot(train_y,'b');
hold on;
plot(signal_pred,'r');
legend({'train\_y','signal\_pred'});
title("matlab forward pass");
hold off;

% [net_updated,signal_pred] = mypredictAndUpdateState(net,train_x);
% 
% figure;
% plot(train_y,'b');
% hold on;
% plot(signal_pred,'r');
% legend({'train\_y','signal\_pred'});
% title("my forward pass");
% hold off;
