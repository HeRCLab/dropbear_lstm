close all;

use_homemade = 1;
epochs=100;

% this script is my sandbox for testing low-level and online training

% signal synthesis
sample_rate = 5000;
time = 2;
points = sample_rate * time;
x = time/points * [1:points]; % for plotting

% signal 1
amplitude = 10;
frequency = 10; % 10 Hz
t = x * 2*pi * frequency;
signal1 = amplitude * sin(t);

% signal 2
amplitude = 4;
frequency = 37;
phase = 1;
t = x * 2*pi * frequency + phase;
signal1 = signal1 + amplitude * sin(t);

% signal 3
amplitude = 7;
frequency = 78;
phase = 2;
t = x * 2*pi * frequency + phase;
signal1 = signal1 + amplitude * sin(t);

% signal 4
amplitude = 2;
frequency = 5; % 10 Hz
t = x * 2*pi * frequency;
signal2 = amplitude * sin(t);

% signal 5
amplitude = 8;
frequency = 67;
phase = .5;
t = x * 2*pi * frequency + phase;
signal2 = signal2 + amplitude * sin(t);

% signal 6
amplitude = 5;
frequency = 75;
phase = 1.1;
t = x * 2*pi * frequency + phase;
signal2 = signal2 + amplitude * sin(t);

%signal = [signal1(1,1:floor(size(signal1,2)/2)),...
%          signal2(1,1:ceil(size(signal2,2)/2))];

% don't switch signals for now
signal = signal1;

% plot original signal
%myfig1 = figure;
%plot (x,signal);

% training parameters
model_sample_rate = 200;
training_window = 40;
prediction_time = 1;
subsample = sample_rate / model_sample_rate;
network_type = 'mlp';
if strcmp(network_type,'lstm')
    numHiddenUnits = 10; % for LSTM
    history_length = 1;
else
    history_length = 10; % for MLP
end

% synthesize subsampled signal
x_sub = x(1:subsample:end);
signal_sub = signal(1:subsample:end);

% plot subsampled signal
myfig2 = subplot(2,1,1);
plot (x_sub,signal_sub);

% plot the training windows
min_val = min(signal_sub);
range = max(signal_sub) - min_val;
num_windows = numel(signal_sub)/training_window;
window_plot = zeros(size(signal_sub));
offset=0;
for i=1:numel(signal_sub)
    window_plot(1,i)=min_val+offset;
    if mod(i,training_window)==0
        offset = offset + range/num_windows;
    end
end
hold on;
plot(x_sub,window_plot);
title('training signal');

% allocate predicted signal
signal_pred = zeros(1,numel(signal_sub));

if strcmp(network_type,'lstm')
    % LSTM network
    layers = [ ...
        sequenceInputLayer(1)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(1)
        regressionLayer];

else
    % MLP network
    layers = [ ...
        imageInputLayer([history_length,1,1]) ...
        fullyConnectedLayer(10) ...
        fullyConnectedLayer(1) ...
        regressionLayer];
end

% training parameters
options = trainingOptions('adam', ...
    'MaxEpochs',epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1);

% iterate through each training window
for i = 1:training_window:numel(signal_sub)-prediction_time-training_window-1
    
    % extract x and y
    start_train = i;
    end_train = i+training_window-1;

    if strcmp(network_type,'lstm')
        x_train = signal_sub(start_train:end_train);
        start_pred = i+prediction_time;
        end_pred = i+training_window+prediction_time-1;
        y_train = signal_sub(start_pred:end_pred);
    else % for mlp
        x_train = [];
        y_train = [];
        for j=1:training_window
            start_idx = i-history_length+j;
            if start_idx < 1
                continue;
            end
            end_idx = i+j-1;
            x_train = [x_train;signal_sub(start_idx:end_idx)];
            y_train = [y_train;signal_sub(end_idx+prediction_time)];
        end
    end

    % train network
    if strcmp(network_type,'mlp')
        
        % check if we want to use our home-made trainer
        if use_homemade
           mynet = build_ann(x_train,y_train,[10],epochs);
        else
            x_train2 = zeros(size(x_train,2),1,1,size(x_train,1));
            for j=1:size(x_train,1)
                x_train2(:,:,:,j) = reshape(x_train(j,1:size(x_train,2)),size(x_train,2),1,1);
            end
            net = trainNetwork(x_train2,y_train,layers,options);
        end
    else
        net = trainNetwork(x_train,y_train,layers,options);
    end

    % predict next window
    
    % first, compute the prediction interval (which would be the next
    % window)
    start_pred_signal = i+training_window+prediction_time;
    end_pred_signal = min(i+2*training_window+prediction_time-1,...
                          numel(signal_pred));
                      
    % clip when the next window is out of bounds
    % TODO: don't bother training the last window, eh?
    clip = i+2*training_window+prediction_time-1 - end_pred_signal;

    start_input = i+training_window;
    end_input = i+2*training_window-1-clip;
    x_input = signal_sub(start_input:end_input);

    if strcmp(network_type,'lstm')
        [net,...
            signal_pred(start_pred_signal:end_pred_signal)] = ...
                        predictAndUpdateState(net,x_input);
    else % mlp
        %x_input = signal_pred(start_pred_signal:end_pred_signal);
        x_input = [];
        for j=1:training_window
            start_idx = start_input-history_length+j;
            end_idx = start_input+j-1;
            if end_idx > size(signal_sub,2)
                break;
            end
            x_input = [x_input;signal_sub(start_idx:end_idx)];
        end
          
        if ~use_homemade
            % this crap is necessitated by MATLAB's asinine requirements
            % on the structure of the input data for MLPs
            x_input2 = zeros(size(x_train,2),1,1,size(x_train,1));
            for j=1:size(x_input,1)
                x_input2(:,:,:,j) = reshape(x_input(j,1:size(x_input,2)),size(x_input,2),1,1);
            end
            temp=predict(net,x_input2)';
        else
            % homemade version of the mlp
            temp=mypredict(mynet,x_input)';
        end

        % in case the predicted window is longer than the remaining
        % vector
        if sum(size(signal_pred(start_pred_signal:end_pred_signal)) ~= size(temp)) == 2
            signal_pred(start_pred_signal:end_pred_signal) = temp;
        else
            remaining = size(signal_pred(start_pred_signal:end_pred_signal),2);
            signal_pred(start_pred_signal:end_pred_signal) = temp(1,1:remaining);
        end
    end

    error = signal_pred(start_pred_signal:end_pred_signal) -...
                signal_sub(start_pred_signal:end_pred_signal);

    rmse = mean(error.^2)^.5;

    fprintf('rms of segment training window %d = %0.4e\n',floor((i-1)/training_window)+1,rmse);

    %figure;
    %plot(x_sub,signal_pred);
    
end

% plot predicted signal
subplot(2,1,2);
plot(x_sub,signal_pred);

% draw zigzgags on non-predicted area
hold on;
offset=0;
val=min_val;
for i=1:training_window+prediction_time-1
    window_plot(1,i)=val;
    val = val + 1;
    if val>min_val+range
        val = min_val;  
    end
end

% draw training_windows
for i=training_window+prediction_time:numel(signal_sub)
    window_plot(1,i)=min_val+offset;
    if mod(i,training_window)==0
        offset = offset + range/num_windows;
    end
end
plot(x_sub,window_plot);

errors = signal_sub(training_window+prediction_time:end)-...
         signal_pred(training_window+prediction_time:end);

rmse = mean(errors.^2)^.5;
str = sprintf('predicted signal, rms = %0.4e',rmse);
title(str);
%ylim([min_val,min_val+range]);
