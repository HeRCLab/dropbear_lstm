% these are the main parameters to adjust
% num_lstm_layers = number of LSTM cells
num_lstm_layers = 2;

% weight_sparsity = the percentage of recurrent weights that should remain
% after sparsification
weight_sparsity = .5;

% units = the number of units per cell, express as a range
units = [50:25:250];

% prediction_time = forecast distance (how many samples into the future do
% you want to predict)
prediction_time = 20;

% Read the drop tower data
droptowerdata = importfile_droptower("drop_tower_data.csv");

% extract time axis
x = droptowerdata.time';

% extract signal
signal = droptowerdata.test1';
    
% infer the time span, sample rate, and start time
time_span = x(end) - x(1);
sample_rate = numel(x)/time_span;
time_offset = x(1);

% seed RNG
rng(42);

% initialize the result arrays (we might not need this)
SNR_model = [];
SNR_subsampling = [];

% specify the sweep parameters.  in this case, 50 to 250 units in
% increments of 25 units.
sweep_points = numel(units);

% initialize dynamic array for results
model_snr = [];

for i=1:sweep_points
    
    % set the number of units for LSTM to be considered
    num_units = units(i);
    
    % number of training samples
    % note that we don't need to substract for units for LSTMs, but
    % this is here to allow this code to work with MLPs too.  it won't affect
    % the LSTM results significantly.
    training_samples = numel(signal)-prediction_time;

    % allocate memory for the training data
    training_batch_x = zeros(training_samples,1);
    training_batch_y = zeros(training_samples,1);

    % allocate and zero-pad predicted signal
    signal_pred = zeros(1,training_samples);

    % set up the LSTM model
    layers = [ sequenceInputLayer(1) ];
    for layer = 1:num_lstm_layers
        layers = [layers lstmLayer(num_units)];
    end
    layers = [layers fullyConnectedLayer(1) regressionLayer];

    % training parameters
    opts = trainingOptions('adam', ...
        'MaxEpochs',epochs, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0);

    % copy training data
    train_x = signal(1:end-prediction_time);
    train_y = signal(prediction_time+1:end);

    % train the LSTM
    net = trainNetwork(train_x,train_y,layers,opts);

    % sparsity recurrent weights
    if weight_sparsity < 1
        net_sparse = sparsify_net(net,weight_sparsity);
    else
        net_sparse = net;
    end

    % generate forecasted network
    [net,signal_pred] = predictAndUpdateState(net_sparse,train_x);

    % phase shift it!
    fill = zeros(1,prediction_time);
    signal_pred = [fill signal_pred];

    % determine SNR of the model
    error_signal = signal - signal_pred;
    % compute error power
    error_power = rms(error_signal)^2;
    % compute signal power
    signal_power = rms(signal)^2;
    % compute SNR
    model_snr = [model_snr,log10(signal_power / error_power) * 20]
end

figure;
plot(x,signal,'b.');
hold on;
plot(x,signal_pred,'r');
legend({'signal','predicted signal'});
xlabel('time (s)');
ylabel('acceleration (kilo-g)');
title("LSTM with "+num_lstm_layers+" layers and "+units(end)+" units per layer with compression = "+weight_sparsity);
hold off;

figure;
plot(units,model_snr,'r');
hold on;
legend({'accuracy'});
xlabel('units per cell');
ylabel('SNR (dB)');
title("LSTM with "+num_lstm_layers+" layers and "+units(end)+" units per layer with compression = "+weight_sparsity);
