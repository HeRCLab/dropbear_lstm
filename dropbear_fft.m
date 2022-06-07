MLP = 0;
LSTM = 1;

window_size = 0.5;
sample_rate = 2000;
epochs = 50;
mlp_hidden_neurons = 50;
lstm_units = 10;

% read data and compute sample rates
data = jsondecode(fileread('data_6_with_FFT.json'));
vibration_sample_rate = numel(data.acceleration_data) / (data.time_acceleration_data(end) - data.time_acceleration_data(1));
pin_sample_rate = numel(data.measured_pin_location) / (data.measured_pin_location_tt(end) - data.measured_pin_location_tt(1));
vibration_signal = data.acceleration_data';
pin_position = data.measured_pin_location';

% remove nans in pin position
for i = find(isnan(pin_position))
    pin_position(i) = (pin_position(i+1) + pin_position(i-1))/2;
end

x = data.time_acceleration_data';

% clip vibration signal due to missing pin samples at start
clip = find(x < data.measured_pin_location_tt(1));
x = x(clip(end):end);
vibration_signal = vibration_signal(clip(end):end);

% subsample vibration data
[x_sub,vibration_signal_sub] = myresample(vibration_signal,vibration_sample_rate,sample_rate);

% resample pin data
% NOTE: this resets the xaxis to start from 0 instead of where the pin
% position data actually starts (~1 s)
[x_sub,pin_position_resamp] = myresample(pin_position,pin_sample_rate,sample_rate);

% clip extra samples
signal_length=min(size(vibration_signal_sub,2),size(pin_position_resamp,2));
vibration_signal_sub = vibration_signal_sub(1,1:signal_length);
x_sub = x_sub(1,1:signal_length);
pin_position_resamp = pin_position_resamp(1,1:signal_length);

% compute fft window size in samples
fft_window = ceil(window_size * pin_sample_rate);

% allocate memory
vibration_signal_sub_fft = zeros(fft_window,size(vibration_signal_sub,2));

% compute FFT
for i=fft_window:size(vibration_signal_sub,2)
    vibration_signal_sub_fft(:,i) = abs(fft(vibration_signal_sub(1,i-fft_window+1:i)));
end

% training options
opts = trainingOptions('sgdm', ...
    'MaxEpochs',epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',true);

% build NN
if MLP
    layers = [imageInputLayer([fft_window,1,1]) fullyConnectedLayer(mlp_hidden_neurons) tanhLayer fullyConnectedLayer(mlp_hidden_neurons) tanhLayer fullyConnectedLayer(1) regressionLayer];
    % allocate training data
    train_x = zeros(fft_window,1,1,size(vibration_signal_sub_fft,2));

    % organize training data
    for i=1:size(train_x,4)
        train_x(:,1,1,i) = vibration_signal_sub_fft(:,i);
    end
    
    net = trainNetwork(train_x,pin_position_resamp',layers,opts);
end

if LSTM
    layers = [sequenceInputLayer(size(vibration_signal_sub_fft,1)) lstmLayer(lstm_units) lstmLayer(lstm_units) fullyConnectedLayer(1) regressionLayer];
    train_x = vibration_signal_sub_fft;
    net = trainNetwork(train_x,pin_position_resamp,layers,opts);
end

%%
pin_position_pred = predict(net,train_x);

figure
hold on;
plot(x_sub,pin_position_resamp,'r');
plot(x_sub,pin_position_pred,'b');
legend({'actual','predicted'});
xlabel('time');

if MLP
    rmse = mean((pin_position_resamp'-pin_position_pred).^2)^.5
end

if LSTM
    rmse = mean((pin_position_resamp-pin_position_pred).^2)^.5
end
