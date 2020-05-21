% this script is my sandbox for testing low-level and online training

% close all the plots from previous run
close all;
clear all;

% clear errors
errors=[];

% initialize network for low-level trainer
first_training=1;

% parameters
% reuse network across trainings; otherwise, start from randomized
% network each training
reuse_network = 1;

% number of hidden neurons
hidden_size=20;

% use homemade (low-level) training algorithm
use_homemade = 1;

% epochs
epochs=1;

% signal synthesis
sample_rate = 5000;
time = 2;
amps=[10,4,7,2,8,5];
freqs=[10,37,78,5,67,75];
phases=[0,0,0,0,0,0];

% synthesize signal
[x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

% training parameters
model_sample_rate = 200;
prediction_time = 10;
subsample = sample_rate / model_sample_rate;

% select network type
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
myfig2 = subplot(3,1,1);
plot(x_sub,signal_sub);
title('subsampled signal');

% % plot the training windows
% min_val = min(signal_sub);
% range = max(signal_sub) - min_val;
% num_windows = numel(signal_sub)/1;
% window_plot = zeros(size(signal_sub));
% offset=0;
% for i=1:numel(signal_sub)
%     window_plot(1,i)=min_val+offset;
%     if mod(i,1)==0
%         offset = offset + range/num_windows;
%     end
% end
% 
% subplot(4,1,2);
% plot(x_sub,window_plot);
% title('training signal');

% allocate predicted signal
signal_pred = zeros(1,numel(signal_sub));

if strcmp(network_type,'lstm')
    % LSTM network
    layers = [ ...
        sequenceInputLayer(1)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(1)
        regressionLayer];
    clear net;
    net.Layers=layers;

else
    % MLP network
    layers = [ ...
        imageInputLayer([history_length,1,1]) ...
        fullyConnectedLayer(hidden_size) ...
        fullyConnectedLayer(1) ...
        regressionLayer];
    clear net;
    net.Layers=layers;
end

% training parameters
options = trainingOptions('adam', ...
    'MaxEpochs',epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.002, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0);

subplot(3,1,2);

% iterate through each training window
for i = 1:1:numel(signal_sub)-prediction_time-1-1
    
    % extract x and y
    start_train = i;
    end_train = i+1-1;
    
    fprintf ("extracting training iterval %d to %d, prediction_time=%d\n",...
                        start_train,end_train,prediction_time);

    if strcmp(network_type,'lstm')
        x_train = signal_sub(start_train:end_train);
        start_pred = i+prediction_time;
        end_pred = i+1+prediction_time-1;
        y_train = signal_sub(start_pred:end_pred);
    else % for mlp
        x_train = [];
        y_train = [];
        for j=1:1
            start_idx = i-history_length+j;
            if start_idx < 1
                continue;
            end
            end_idx = i+j-1;
            x_train = [x_train;signal_sub(start_idx:end_idx)];
            y_train = [y_train;signal_sub(end_idx+prediction_time)];
            
            fprintf ("  training sample iterval %d to %d, history_length=%d\n",...
                        start_idx,end_idx,history_length);
        end
    end

    if isempty(x_train)
        continue;
    end
    
    % train network
    if strcmp(network_type,'mlp')
        
        % check if we want to use our home-made trainer
        if use_homemade
           if first_training
               mynet = build_ann(x_train,y_train,[10],epochs);
               first_training=0;
           else
               mynet = build_ann(x_train,y_train,[10],epochs,mynet);
           end
        else
            % convert the training data into the necessary format for
            % Matlab
            x_train2 = zeros(size(x_train,2),1,1,size(x_train,1));
            for j=1:size(x_train,1)
                x_train2(:,:,:,j) = reshape(x_train(j,1:size(x_train,2)),size(x_train,2),1,1);
            end
            
            % train network
            if reuse_network
                net = trainNetwork(x_train2,y_train,net.Layers,options);
            else
                net = trainNetwork(x_train2,y_train,layers,options);
            end
        end
    else
        if reuse_network
            net = trainNetwork(x_train2,y_train,net,options);
        else
            net = trainNetwork(x_train2,y_train,layers,options);
        end
    end

    % predict next window
    
    % first, compute the prediction interval (which would be the next
    % window)
    start_pred_signal = i+prediction_time;
    end_pred_signal = min(i+prediction_time,...
                          numel(signal_pred));
                      
    start_input = i-prediction_time+1;
    end_input = i+1-1;
    x_input = signal_sub(start_input:end_input);

    if strcmp(network_type,'lstm')
        [net,...
            signal_pred(start_pred_signal:end_pred_signal)] = ...
                        predictAndUpdateState(net,x_input);
    else % mlp
        %x_input = signal_pred(start_pred_signal:end_pred_signal);
        x_input = [];
        for j=1:1
            start_idx = start_input+j-1;
            end_idx = start_input+history_length+j-2;
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

    fprintf('rms of segment training window %d = %0.4e\n',floor((i-1)/1)+1,rmse);
    errors=[errors,rmse];
    subplot(3,1,2);
    plot(errors,'r');
    title('RMS error');
    xlabel('sample number');
    ylabel('rmse');
    pause(1e-6);
    
    %figure;
    %plot(x_sub,signal_pred);
    
end

% plot predicted signal
subplot(3,1,3);
plot(x_sub,signal_pred);

% draw zigzgags on non-predicted area
% offset=0;
% val=min_val;
% for i=1:1+prediction_time-1
%     window_plot(1,i)=val;
%     val = val + 1;
%     if val>min_val+range
%         val = min_val;  
%     end
% end

errors = signal_sub(1+prediction_time:end)-...
         signal_pred(1+prediction_time:end);

rmse = mean(errors.^2)^.5;
str = sprintf('predicted signal, rms = %0.4e',rmse);
title(str);
%ylim([min_val,min_val+range]);
