% parameters

% choose a network
MLP = 0;
LSTM = 1;

% add an FFT front-end?
use_fft = 0;

% if MLP, choose MLP hidden layer size and number of hidden layers
mlp_hidden_neurons = 1000;
num_mlp_hidden_layers = 5;

% if LSTM, choose units/cell and number of cells
lstm_units = 10;
num_lstm_cells = 5;

% if LSTM, choose whether to use built-in or hand-written forward pass code
use_my_predict = 1;

% choose a portion of the signal on which to train
train_start = 0;
train_end = 60;

% if so, choose a window size for FFT
window_size = .1; % seconds

% choose a sample rate
sample_rate = 400;

% choose training time
epochs = 10;

%%
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

% extract training portion
start_sample = find(x_sub>=train_start);
start_sample = start_sample(1);
end_sample = find(x_sub<train_end);
end_sample = end_sample(end);
x_sub_train = x_sub(start_sample:end_sample);
vibration_signal_sub_train = vibration_signal_sub(start_sample:end_sample);
pin_position_resamp_train = pin_position_resamp(start_sample:end_sample);

if use_fft

    % compute fft window size in samples
    fft_window = ceil(window_size * sample_rate);

    % allocate memory
    vibration_signal_sub_fft = zeros(fft_window,size(vibration_signal_sub,2));

    % compute FFT (full dataset)
    for i=fft_window:size(vibration_signal_sub,2)
        vibration_signal_sub_fft(:,i) = abs(fft(vibration_signal_sub(1,i-fft_window+1:i)));
    end

    vibration_signal_sub_fft_train = vibration_signal_sub_fft(:,start_sample:end_sample);
    
end

% training options
opts = trainingOptions('sgdm', ...
    'MaxEpochs',epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle', 'every-epoch', ...
    'LearnRateDropPeriod',200, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',true);

% build NN
if MLP
    
    layers = [imageInputLayer([fft_window,1,1])];
    
    for i=1:num_mlp_hidden_layers
        layers = [layers fullyConnectedLayer(mlp_hidden_neurons) tanhLayer];
    end
    
    layers = [layers fullyConnectedLayer(1) regressionLayer];
    
    % allocate training data
    train_x = zeros(fft_window,1,1,size(vibration_signal_sub_fft_train,2));

    % organize training data
    for i=1:size(train_x,4)
        if use_fft
            train_x(:,1,1,i) = vibration_signal_sub_fft_train(:,i);
        else
            train_x(:,1,1,i) = vibration_signal_sub(:,i);
        end
    end
    
    net = trainNetwork(train_x,pin_position_resamp_train',layers,opts);
end

if LSTM
    if use_fft
        layers = [sequenceInputLayer(size(vibration_signal_sub_fft_train,1))];
    else
        layers = [sequenceInputLayer(size(vibration_signal_sub,1))];
    end
    
    for i=1:num_lstm_cells
        layers = [layers lstmLayer(lstm_units)];
    end
        
    layers = [layers fullyConnectedLayer(1) regressionLayer];
    
    if use_fft
        train_x = vibration_signal_sub_fft_train;
    else
        train_x = vibration_signal_sub;
    end
    net = trainNetwork(train_x,pin_position_resamp_train,layers,opts);
end

%%

% predict training data
if LSTM && use_my_predict
    [net,pin_position_pred_train,cell_states,hidden_states] = mypredictAndUpdateState2(net,train_x);
else
    pin_position_pred_train = predict(net,train_x);
end

% plot train
figure
hold on;
plot(x_sub_train,pin_position_resamp_train,'r');
plot(x_sub_train,pin_position_pred_train,'b');
legend({'actual','predicted'});
xlabel('time');

% repackage data to predict full dataset
if MLP
    if use_fft
        % allocate training data
        test_x = zeros(fft_window,1,1,size(vibration_signal_sub_fft,2));

        % organize training data
        for i=1:size(test_x,4)
            test_x(:,1,1,i) = vibration_signal_sub_fft(:,i);
        end
    else
        % allocate training data
        test_x = zeros(fft_window,1,1,size(vibration_signal_sub,2));

        % organize training data
        for i=1:size(test_x,4)
            test_x(:,1,1,i) = vibration_signal_sub(:,i);
        end
    end
end

if LSTM
    if use_fft
        test_x = vibration_signal_sub_fft;
    else
        test_x = vibration_signal_sub;
    end
end

% predict full data
pin_position_pred = predict(net,test_x);

% plot full
figure
hold on;
plot(x_sub,pin_position_resamp,'r');
plot(x_sub,pin_position_pred,'b');
legend({'actual','predicted'});
xlabel('time');

%%
if MLP
    rmse_full = mean((pin_position_resamp_train'-pin_position_pred_train).^2)^.5
    rmse_full = mean((pin_position_resamp'-pin_position_pred).^2)^.5
end

if LSTM
    rmse_full = mean((pin_position_resamp_train-pin_position_pred_train).^2)^.5
    rmse_full = mean((pin_position_resamp-pin_position_pred).^2)^.5
end
