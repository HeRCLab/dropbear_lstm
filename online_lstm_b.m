close all;

% read Puga's signal
data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
x = data.Segment1.data(:,1);
signal = data.Segment1.data(:,4)';
sample_rate = numel(data.Segment1.data(:,1))/data.Segment1.data(end,1);

% signal synthesis
sample_rate = 5000;
time = 10;
amps=[1, 2, 3, 0.7, 2.3, 1];
freqs=[2, 3.7, 7.80, 0.12, 0.54, 1.3];
phases=[0, 1, 2, 3, 5, 1];
% synthesize signal
[x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

training_window = numel(signal);
prediction_time = 1;

rmse=[];
sample_rates=[];

% training parameters
for downsample = 1 : -.01 : .01
    
    model_sample_rate = sample_rate * downsample;
    sample_rates = [sample_rates,model_sample_rate];
    subsample = floor(sample_rate / model_sample_rate);
    network_type = 'mlp';
    history_length = ceil(.01 * model_sample_rate); % 10 ms

    % synthesize subsampled signal
    x_sub = x(1:subsample:end);
    signal_sub = signal(1:subsample:end);

    % plot subsampled signal
    myfig2 = subplot(2,1,1);
    plot (x_sub,signal_sub);

    % allocate predicted signal
    signal_pred = zeros(1,numel(signal_sub));

    % MLP network
    layers = [ ...
        imageInputLayer([history_length,1,1]) ...
        fullyConnectedLayer(10) ...
        fullyConnectedLayer(1) ...
        regressionLayer];

    % training parameters
    options = trainingOptions('adam', ...
        'MaxEpochs',5, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',true...
        );

    % extract x and y
    start_train = 1;
    end_train = numel(signal_sub);

    x_train = [];
    y_train = [];
    for j=1:numel(signal_sub)-history_length
        start_idx = j;
        end_idx = j+history_length-1;
        x_train = [x_train;signal_sub(start_idx:end_idx)];
        y_train = [y_train;signal_sub(end_idx+prediction_time)];
    end
    
    % train network
    x_train2 = zeros(size(x_train,2),1,1,size(x_train,1));
    for j=1:size(x_train,1)
        x_train2(:,:,:,j) = reshape(x_train(j,1:size(x_train,2)),size(x_train,2),1,1);
    end
    net = trainNetwork(x_train2,y_train,layers,options);

    % predict
    error=[];
    for k = history_length+prediction_time:numel(signal_sub)
        x_input2 = reshape(signal_sub(1,k-history_length-prediction_time+1:k-1),size(x_train,2),1,1);
        signal_pred(k) = predict(net,x_input2);
        error = [error,signal_pred(k) - signal_sub(k)];
    end
    rmse = [rmse,mean(error.^2)^.5];

end

plot(sample_rates,rmse);
title('model rmse vs sample rate');
xlabel('sample rate');
ylabel('rmse');
