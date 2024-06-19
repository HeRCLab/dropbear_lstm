% parameters

%gpuDevice(1);

% choose a network
MLP = 0;
LSTM = 1;

% add an FFT front-end?
use_fft = 0;

% if MLP, choose MLP hidden layer size and number of hidden layers
mlp_hidden_neurons = 1000;
num_mlp_hidden_layers = 3;

% if LSTM, choose units/cell and number of cells
lstm_units = [100];
num_lstm_cells = 1;

% if LSTM, choose other training options
training_snippet_size = 50; % seconds
number_of_sequence_inputs = 20; % assuming no FFT
number_of_training_rounds = 1; % number of passes over whole dataset
use_higher_sample_rate_for_inputs = 0; % decouple input and output T_s

% if LSTM, choose whether to use built-in or hand-written forward pass code
use_my_predict = 1;

% choose a portion of the signal on which to train
% note this different from training snippet size
train_start = 0; % seconds
train_end = 60; % seconds

% FFT window size
window_size = .1; % seconds

% choose an output sample rate
sample_rate = 1000;

% choose training time
epochs = 500;

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
else

    vibration_signal_sub_fft_train = vibration_signal_sub;
    %fft_window = 512; % input size, since we're not using an FFT in this configuration

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
    'InitialLearnRate',3e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',200, ...
    'LearnRateDropFactor',0.1, ...
    'Verbose',true,...
    'Minibatchsize',10); % minibatchsize has no effect, since LSTMs are limited to batch size of 1((

%%

% build NN
if MLP
    layers = [imageInputLayer([number_of_sequence_inputs,1,1])];
    
    layers = [layers ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2) ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2) ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2) ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2) ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2) ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2) ...
        convolution2dLayer(3,256) reluLayer maxPooling1dLayer(2)];

    for i=1:num_mlp_hidden_layers
        layers = [layers fullyConnectedLayer(mlp_hidden_neurons) tanhLayer];
    end
    
    layers = [layers fullyConnectedLayer(1) regressionLayer];
    
    % allocate training data
    if use_fft
        train_x = zeros(fft_window,1,1,size(vibration_signal_sub_fft_train,2));
    else
        train_x = zeros(number_of_sequence_inputs,size(vibration_signal_sub,2));
    end

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
%%
dump_weights(net);
%%
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
    clip_start = find(data.time_acceleration_data>latest_start_time);
    clip_end = find(data.time_acceleration_data<earliest_end_time);
    if isempty(clip_end)
        clip_end = numel(data.time_acceleration_data);
    end
    data.time_acceleration_data = data.time_acceleration_data(clip_start(1):clip_end(end));
    data.acceleration_data = data.acceleration_data(clip_start(1):clip_end(end));
    
    clip_start = find(data.measured_pin_location_tt>latest_start_time);
    clip_end = find(data.measured_pin_location_tt<earliest_end_time);
    if isempty(clip_end)
        clip_end = numel(data.measured_pin_location_tt);
    end
    data.measured_pin_location_tt = data.measured_pin_location_tt(clip_start(1):clip_end(end));
    data.measured_pin_location = data.measured_pin_location(clip_start(1):clip_end(end));
    
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

    if (time_pin(1) < time_vibration(1))
        time_pin = time_vibration;
    else
        time_vibration = time_pin;
    end
    
    % interpolate signals
    vibration_signal = interp1(data.time_acceleration_data,data.acceleration_data,time_vibration);
    pin_position = interp1(data.measured_pin_location_tt,data.measured_pin_location,time_vibration);

    myfile = fopen('dropbear.dat','w');

    % write to file
    for i=1:numel(vibration_signal)
        fwrite(myfile,vibration_signal(i),'single');
    end

    for i=1:numel(pin_position)
        fwrite(myfile,pin_position(i),'single');
    end

    for i=1:numel(time_vibration)
        fwrite(myfile,time_vibration(i),'single');
    end

    fclose(myfile);
end

function [input_gate,forget_gate,output_gate,modulation_gate] = extract_weights (layer)
    % (units * 4) x 1
    input_weights = layer.InputWeights;
    % (units * 4) x units
    recurrent_weights = layer.RecurrentWeights;
    % (units * 4) x 1
    bias = layer.Bias;
    % infer the number of units
    num_units = size(input_weights,1)/4;
    % get values for current gate (since they are packed)
    
    perms = [1 0 3 2];

    segment = perms(1);
    chunk = num_units*segment+1:num_units*(segment+1);
    forget_gate.input_weights = input_weights(chunk,:);
    forget_gate.recurrent_weights = recurrent_weights(chunk,:);
    forget_gate.bias = bias(chunk,1);
    
    segment = perms(2);
    chunk = num_units*segment+1:num_units*(segment+1);
    input_gate.input_weights = input_weights(chunk,:);
    input_gate.recurrent_weights = recurrent_weights(chunk,:);
    input_gate.bias = bias(chunk,1);

    segment = perms(3);
    chunk = num_units*segment+1:num_units*(segment+1);
    output_gate.input_weights = input_weights(chunk,:);
    output_gate.recurrent_weights = recurrent_weights(chunk,:);
    output_gate.bias = bias(chunk,1);

    segment = perms(4);
    chunk = num_units*segment+1:num_units*(segment+1);
    modulation_gate.input_weights = input_weights(chunk,:);
    modulation_gate.recurrent_weights = recurrent_weights(chunk,:);
    modulation_gate.bias = bias(chunk,1);
end

function [] = write_weights_to_file (myfile,recurrent_weights,input_weights,bias)

    for i=1:size(recurrent_weights,1)
        for j=1:size(recurrent_weights,2)
            fwrite(myfile,recurrent_weights(i,j),'single');
        end
    end

    for i=1:size(input_weights,1)
        for j=1:size(input_weights,2)
            fwrite(myfile,input_weights(i,j),'single');
        end
    end

    for i=1:numel(bias)
        fwrite(myfile,bias(i),'single');
    end
end

function [] = dump_weights (net)

    % allocate hidden and cell states for all LSTM layers
	for i = 1:size(net.Layers,1)
        layer = net.Layers(i);
        if strcmp(class(layer),'nnet.cnn.layer.LSTMLayer')
            [input_gate,forget_gate,output_gate,modulation_gate] = extract_weights (layer);
        elseif strcmp(class(layer),'nnet.cnn.layer.FullyConnectedLayer')
            fully_connected_weights = layer.Weights;
            fully_connected_bias = layer.Bias;
        end
    end

    % write to file
    myfile=fopen("weights.dat","w+");

    write_weights_to_file (myfile,input_gate.recurrent_weights,input_gate.input_weights,input_gate.bias);
    write_weights_to_file (myfile,forget_gate.recurrent_weights,forget_gate.input_weights,forget_gate.bias);
    write_weights_to_file (myfile,output_gate.recurrent_weights,output_gate.input_weights,output_gate.bias);
    write_weights_to_file (myfile,modulation_gate.recurrent_weights,modulation_gate.input_weights,modulation_gate.bias);

    fclose(myfile);
end
