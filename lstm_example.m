% get droptower data
droptowerdata = importfile_droptower("drop_tower_data.csv");
x = droptowerdata.time';
signal = droptowerdata.test1';

% set up the time axis
time_span = x(end) - x(1);
sample_rate = numel(x)/time_span;
time_offset = x(1);

% create LSTM
layers = [sequenceInputLayer(1) lstmLayer(50) lstmLayer(50) fullyConnectedLayer(1) regressionLayer];
layers = [sequenceInputLayer(1) lstmLayer(400) fullyConnectedLayer(1) regressionLayer];

% train it
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

% predict using both the custom and the built-in forward pass

% restore original state of trained network
net2=net;

% use our function to perform forward pass
[net_updated,signal_pred] = mypredictAndUpdateState2(net2,train_x);

% compute SNR
error_power = rms(signal_pred - signal(1,20:end));
signal_power = rms(signal);
snr = log10(signal_power/error_power)*20
close all
plot(train_y,'b');
hold on;
plot(signal_pred,'r');

% restore original state of trained network
net2=net;

% use the Matlab function to perform forward pass
[net_updated,signal_pred] = predictAndUpdateState(net2,train_x);
plot(signal_pred,'g');
legend({'train\_y','mypredictAndUpdateState()','predictAndUpdateState()'});
