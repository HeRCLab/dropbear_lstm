% parameters

% choose a network
MLP = 0;
LSTM = 1;

% add an FFT front-end?
use_fft = 0;

% if MLP, choose MLP hidden layer size and number of hidden layers
mlp_hidden_neurons = 1000;
num_mlp_hidden_layers = 3;

% if LSTM, choose units/cell and number of cells
lstm_units = [20,20,20];
num_lstm_cells = 3;

% if LSTM, choose other training options
training_snippet_size = 50; % seconds
number_of_sequence_inputs = 20; % assuming no FFT
number_of_training_rounds = 1; % number of passes over whole dataset
use_higher_sample_rate_for_inputs = 1; % decouple input and output T_s

% if LSTM, choose whether to use built-in or hand-written forward pass code
use_my_predict = 0;

% choose a portion of the signal on which to train
% note this different from training snippet size
train_start = 0; % seconds
train_end = 60; % seconds

% FFT window size
window_size = .1; % seconds

% choose an output sample rate
sample_rate = 2000;

% choose training time
epochs = 100;

[time_vibration,vibration_signal,...
    time_pin,pin_position] = read_and_clean_dataset('data_6_with_FFT.json', ...
                                                    sample_rate,...
                                                    number_of_sequence_inputs, ...
                                                    use_higher_sample_rate_for_inputs);

% rename for the benefit of legacy code
x_sub_vib = time_vibration;
vibration_signal_sub = vibration_signal;
x_sub_pin = time_pin;
pin_position_resamp = pin_position;

% extract training portion (vibration)
start_sample_vib = find(x_sub_vib>=train_start);
start_sample_vib = start_sample_vib(1);
end_sample_vib = find(x_sub_vib<train_end);
end_sample_vib = end_sample_vib(end);
x_sub_train_vib = x_sub_vib(start_sample_vib:end_sample_vib);
vibration_signal_sub_train = vibration_signal_sub(start_sample_vib:end_sample_vib);

% extract training portion (pin)
start_sample_pin = find(x_sub_pin>=train_start);
start_sample_pin = start_sample_pin(1);
end_sample_pin = find(x_sub_pin<train_end);
end_sample_pin = end_sample_pin(end);
x_sub_train_pin = x_sub_pin(start_sample_pin:end_sample_pin);
pin_position_resamp_train = pin_position_resamp(start_sample_pin:end_sample_pin);

% plot for sanity check
figure;
hold on;
title('training data');
plot(x_sub_train_vib,vibration_signal_sub_train,'g');
yyaxis right
plot(x_sub_train_pin,pin_position_resamp_train,'r');
title('training data');
legend({"vibration","pin position"});
xlabel('time (s)');
hold off;
drawnow;

%%
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

% collate data if needed
if number_of_sequence_inputs > 1
    if ~use_higher_sample_rate_for_inputs
        for i=1:(number_of_sequence_inputs-1)
            vibration_signal_sub = [vibration_signal_sub;...
                                    zeros(1,i) vibration_signal_sub(1,1:end-i)];
        end
    else
        round_down = floor(numel(vibration_signal_sub)/number_of_sequence_inputs)*number_of_sequence_inputs;
        vibration_signal_sub = reshape(vibration_signal_sub(1,1:round_down),...
                                         number_of_sequence_inputs,...
                                         []);
    end
end

% training options
opts = trainingOptions('adam', ...
    'MaxEpochs',epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',3e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',200, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',true,...
    'Minibatchsize',20); % minibatchsize has no effect, since LSTMs are limited to batch size of 1((

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
    
    for i=1:numel(lstm_units)
        layers = [layers lstmLayer(lstm_units(i))];
    end
        
    layers = [layers fullyConnectedLayer(1) regressionLayer];
    
    if use_fft
        train_x = vibration_signal_sub_fft_train;
    else
        train_x = vibration_signal_sub;
    end
    
    % perform multiple passes to train this network (experimental)
    % for now, assume that the chunks are not overlapping (TODO: try
    % overlapping)
    
    figure;
    hold on;
    plot(x_sub_train_pin,pin_position_resamp_train,'r');
    xlabel('time (s)');
    
    % plot colors
    colors = hsv(number_of_training_rounds+1);
    
    data_duration = size(train_x,2)/sample_rate;
    number_of_chunks = ceil(data_duration/training_snippet_size);
    chunk_size = floor(size(train_x,2) / number_of_chunks);
    lineobjs = cell(1,number_of_chunks);
    for round=1:number_of_training_rounds
        for chunk=1:number_of_chunks
            index_range = (chunk-1)*chunk_size+1:chunk*chunk_size;
            fprintf("training chunk %d/%d (%d/%d samples)\n",chunk,number_of_chunks,numel(index_range),size(train_x,2));

            if chunk==1
                net = trainNetwork(train_x(:,index_range),pin_position_resamp_train(:,index_range),layers,opts);
            else
                net = trainNetwork(train_x(:,index_range),pin_position_resamp_train(:,index_range),layerGraph(net.Layers),opts);
            end

            %if ~isempty(lineobjs{1,chunk}) ~= 0
            %    delete(lineobjs(chunk));
            %end
            preddata = predict(net,train_x(:,index_range));
            plotobj =...
                plot(x_sub_train_pin(1,index_range),preddata,'color',colors(round+1,:));
            drawnow;
            lineobjs{1,chunk} = plotobj;
        end
    end
end

%exportONNXNetwork(net,"dropbear_lstm.onnx");

% predict training data
if LSTM && use_my_predict
    [net,pin_position_pred_train,cell_states,hidden_states] = mypredictAndUpdateState2(net,train_x);
else
    pin_position_pred_train = predict(net,train_x);
end

% plot train
figure
hold on;
plot(x_sub_train_pin(1:size(pin_position_pred_train,2)),pin_position_resamp_train(1:size(pin_position_pred_train,2)),'r');
plot(x_sub_train_pin(1:size(pin_position_pred_train,2)),pin_position_pred_train,'b');
legend({'actual','predicted'});
xlabel('time (s)');

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

%%
% plot full
figure
hold on;
title('predicted data');
plot(x_sub_pin(1:size(pin_position_pred,2)),pin_position_resamp(1:size(pin_position_pred,2)),'r');
plot(x_sub_pin(1:size(pin_position_pred,2)),pin_position_pred,'b');
legend({'actual','predicted'});
xlabel('time');

%%
if MLP
    rmse_full = mean((pin_position_resamp_train'-pin_position_pred_train).^2)^.5
    rmse_full = mean((pin_position_resamp'-pin_position_pred).^2)^.5
end

if LSTM
    rmse_full = mean((pin_position_resamp_train(1:size(pin_position_pred_train,2))-pin_position_pred_train(1:size(pin_position_pred_train,2))).^2)^.5
    rmse_full = mean((pin_position_resamp(1:size(pin_position_pred_train,2))-pin_position_pred).^2)^.5
end


function [time_vibration,vibration_signal,...
    time_pin,pin_position] = read_and_clean_dataset(filename,...
                                                    sample_rate,...
                                                    number_of_sequence_inputs, ...
                                                    use_higher_sample_rate_for_inputs)
    % read data and compute sample rates
    data = jsondecode(fileread(filename));
    
    % native sample rates computed here
    %vibration_sample_rate = numel(data.acceleration_data) / (data.time_acceleration_data(end) - data.time_acceleration_data(1));
    %pin_sample_rate = numel(data.measured_pin_location) / (data.measured_pin_location_tt(end) - data.measured_pin_location_tt(1));
    
    % remove nans in pin position
    for i = find(isnan(data.measured_pin_location))
        data.measured_pin_location(i) = (data.measured_pin_location(i+1) + data.measured_pin_location(i-1))/2;
    end
    
    % determine overlapping time span for both signals
    latest_start_time = max([data.time_acceleration_data(1) data.measured_pin_location_tt(1)]);
    earliest_end_time = min([data.time_acceleration_data(end) data.measured_pin_location_tt(end)]);
    
    % trim signals
    clip_start = find(data.time_acceleration_data>=latest_start_time);
    clip_end = find(data.time_acceleration_data>=earliest_end_time);
    data.time_acceleration_data = data.time_acceleration_data(clip_start(1):clip_end(1));
    data.acceleration_data = data.acceleration_data(clip_start(1):clip_end(1));
    
    clip_start = find(data.measured_pin_location_tt>=latest_start_time);
    clip_end = find(data.measured_pin_location_tt>=earliest_end_time);
    data.measured_pin_location_tt = data.measured_pin_location_tt(clip_start(1):clip_end(1));
    data.measured_pin_location = data.measured_pin_location(clip_start(1):clip_end(1));
    
    % create new time axes
    if use_higher_sample_rate_for_inputs
        sample_rate_vib = sample_rate*number_of_sequence_inputs;
    else
        sample_rate_vib = sample_rate;
    end
    time_vibration = [data.time_acceleration_data(1):...
                        1/sample_rate_vib:...
                        data.time_acceleration_data(end)];
    
    time_pin = [data.measured_pin_location_tt(1):...
                        1/sample_rate:...
                        data.measured_pin_location_tt(end)];
    
    % interpolate signals
    vibration_signal = interp1(data.time_acceleration_data,data.acceleration_data,time_vibration);
    pin_position = interp1(data.measured_pin_location_tt,data.measured_pin_location,time_pin);
end