% this script is my sandbox for testing low-level and online training

% close all the plots from previous run
close all;
clear all;
profile on

% parameters
online_training = 0;
history_length = 40;
hidden_size = 10;
model_sample_rate = 1250;
prediction_time = 10;
alpha = 1e-2;
epochs = 10;
lstm = 0;

% data format
fixed = 0;
wordsize = 4;
fractionsize = 3;

% % signal synthesis
% sample_rate = 5000;
% time = 60;
% amps=[1, 2, 3, 0.7, 2.3, 1];
% freqs=[2, 3.7, 7.80, 0.12, 0.54, 1.3];
% phases=[0, 1, 2, 3, 5, 1];

% % synthesize signal
% [x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

% read input signal
%data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
data=lvm_import('Ivol_Acc_Load_data1_w3_NSTD.lvm');
x = data.Segment1.data(:,1);
signal = data.Segment1.data(:,4)';

% Drop Tower data
droptowerdata = importfile_droptower("drop_tower_data.csv");
x = droptowerdata.time;
signal = droptowerdata.test1;

time_span = x(end) - x(1);
sample_rate = numel(x)/time_span;
time_offset = x(1);

%sample_period = 1/model_sample_rate;
%subsample = floor(sample_rate / model_sample_rate);
%[x_sub,signal_sub] = myresample(signal,sample_rate,model_sample_rate);
% don't subsample drop tower data!
x_sub=x';
signal_sub=signal';

% seed RNG
rng(42);

% debug
global deltas_output;
deltas_output=[];

% clear errors
errors=[];

% initialize empty frames
frames=[];

% initialize network for low-level trainer
first_training=1;

% use homemade (low-level) training algorithm
use_homemade = 1;

myfig2 = figure('Position', [20 50 1650 800]);
hold on;

% assume i is the current sample being read from the accelerometer, so
% we are predicting sample:
%   i+prediction_time
% based on:
% samples i-history_length-prediction_time to i-prediction_time

signal_pred = zeros(size(signal_sub));

training_samples = numel(signal_sub)+history_length;
training_batch_x = zeros(training_samples,history_length);
training_batch_y = zeros(training_samples,1);

% set up plots

% horizontal axis
subplot(3,1,1);
hold on;
xlabel('time (s)');
title('subsampled signal, predicted signal');

subplot(3,1,2);
title('instantaneous error with shifted prediction');
xlabel('time (s)');
ylabel('difference');

subplot(3,1,3);
title("cumulative rms error after shifting predicted signal "+prediction_time+" samples");
xlabel('time (s)');
ylabel('rmse');

% allocate and zero-pad predicted signal
signal_pred = zeros(1,numel(signal_sub));

if fixed==1
    signal_pred = fi(signal_pred,1,wordsize,fractionsize);
end

for i = 1:numel(signal_sub)
    
    if online_training
        % train one sample at a time
        x_train = signal_sub(i:i+history_length-1);
        y_train = signal_sub(history_length+prediction_time-1);
        
        % train network
        if first_training
            [mynet,output_from_mlp] = build_ann(x_train,y_train,[hidden_size],epochs,alpha);
            first_training=0;
        else
            [mynet,output_from_mlp] = build_ann(x_train,y_train,[hidden_size],epochs,alpha,mynet);
        end
    
        x_pred = signal_sub(i-history_length+1:i);
        signal_pred(1,i+history_length+prediction_time-1)=mypredict(mynet,x_pred);
        
        % handle all the plotting
        if mod(i,50)==0
            % plot predicted signal in real-time
            time_axis = 0:(size(p,2)-1);
            time_axis = time_axis ./ model_sample_rate;

            subplot(4,1,1);
            p=signal_pred(prediction_time:i-1);
            s=signal_sub(1:i-prediction_time);
            plot(time_axis,s,'r');
            hold on;
            plot(time_axis,p,'b');

            subplot(4,1,2);
            inst_error = s-p;
            plot(time_axis,inst_error,'r');

            subplot(4,1,3);
            time_axis = 0:(size(rms_error_aligned,2)-1);
            time_axis = time_axis ./ model_sample_rate;
            p=p(history_length+1:end);
            s=s(history_length+1:end);
            rmse = mean((p-s).^2)^.5;
            rms_error_aligned=[rms_error_aligned,rmse];
            plot(time_axis,rms_error_aligned,'r');

            subplot(4,1,4);
            plot(deltas_output);

            drawnow;
        end
        
    else
        % offline training, so only set up training data
        if i+history_length-1 <= numel(signal_sub)
            % input
            training_batch_x(i,:) =...
                signal_sub(i:i+history_length-1);
        
            % expected output
            if i+history_length+prediction_time-1 <= numel(signal_sub)
                training_batch_y(i,1) =...
                    signal_sub(i+history_length+prediction_time-1);
            else
                training_batch_y(i,1) = 0;
            end
        end

    end
    
end

if ~online_training
    % need to clip the predicted values, since we don't know the values
    % past the end of the input file
    training_batch_x = training_batch_x(1:end-prediction_time,:);
    training_batch_y = training_batch_y(1:end-prediction_time,:);
    
    if lstm==0
        % use traditional training for now
        [build_ann,pred,layers] = build_ann (training_batch_x,training_batch_y,[hidden_size],epochs,alpha);
    else
        layers = [ ...
            sequenceInputLayer(1)
            lstmLayer(history_length)
            fullyConnectedLayer(1)
            regressionLayer];
    end

     % training parameters
    opts = trainingOptions('adam', ...
        'MaxEpochs',epochs, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0);

    % trasform x-data into the format that Matlab's training function
    % expects
    if lstm == 0
        train_x = zeros(size(training_batch_x,2),1,1,size(training_batch_x,1));
        for i=1:size(training_batch_x,1)
            train_x(:,1,1,i) = training_batch_x(i,:);
        end
    else
        for i=1:size(training_batch_x,1)
            train_x(i,1) = training_batch_x(i,1);
        end
        train_x = train_x';
        training_batch_y = training_batch_y';
    end

    net = trainNetwork(train_x,training_batch_y,layers,opts);

    % predict

    if fixed==1
       mynet.weights_hidden = fi(net.Layers(2,1).Weights,1,wordsize,fractionsize,'OverflowAction','Wrap');
       mynet.bias_hidden = fi(net.Layers(2,1).Bias,1,wordsize,fractionsize,'OverflowAction','Wrap');
       mynet.weights_output = fi(net.Layers(3,1).Weights,1,wordsize,fractionsize,'OverflowAction','Wrap');
       mynet.bias_output = fi(net.Layers(3,1).Bias,1,wordsize,fractionsize,'OverflowAction','Wrap');
    end

    if lstm == 1
        [net,signal_pred] = predictAndUpdateState(net,train_x);
    else
        % MLP
        input_val = zeros(history_length,1,1);
        for i=1:size(signal_sub,2)-history_length-prediction_time+1
            input_val(:,1,1) = signal_sub(i:i+history_length-1);
            if fixed == 0
                signal_pred(i+history_length+prediction_time-1) = predict(net,input_val);
            else
                input_val = fi(input_val,1,wordsize,fractionsize);
                out_val = mypredict2(mynet,input_val);
                signal_pred(i+history_length+prediction_time-1) = out_val;
            end
        end

    end
        
    % plot
    time_axis = 0:(size(signal_sub,2)-1);
    time_axis = time_axis ./ model_sample_rate + time_offset;

    subplot(3,1,1);
    plot(time_axis,signal_pred(1:size(time_axis,2)),'r');
    hold on;
    plot(time_axis,signal_sub,'b');
    legend ('signal (actual)','signal (predicted)','Location','northwest');

    subplot(3,1,2);
    inst_error = signal_pred(1:size(signal_sub,2))-signal_sub;
    plot(time_axis,inst_error,'r');

    rmse = zeros(1,size(time_axis,2));
    for i=size(signal_sub,2)
        rmse(1,i) = mean((double(signal_pred(1,1:i)) - signal_sub(1,1:i)).^2)^.5;
    end
    subplot(3,1,3);
    plot(time_axis,rmse(1,1:size(time_axis,2)),'r');
    rmse(1,end)
    drawnow;

end

% writerObj = VideoWriter('myVideo.mp4','MPEG-4');%VideoCompressionMethod,'H.264');
% writerObj.Quality=100;
% writerObj.FrameRate = 10;
% open(writerObj);
% for i=1:length(frames)
%     % convert the image to a frame
%     frame = frames(i) ;    
%     writeVideo(writerObj, frame);
% end
% close(writerObj);

