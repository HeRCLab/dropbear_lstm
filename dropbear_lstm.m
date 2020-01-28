close all;
clear all;

% measure the storage, throughput, and accuracy given by different
% frequencies and history length
downsample_levels = 8:8:56; % relative to input sample rate
history_lengths = 10:-1:1;     % relative to 25 samples at input sample rate
   
% allocate measurement data
storage = zeros(numel(downsample_levels),numel(history_lengths));
throughput = zeros(numel(downsample_levels),numel(history_lengths));
rmse_res = zeros(numel(downsample_levels),numel(history_lengths));

% read data
data = jsondecode(fileread('data_6_with_FFT.json'));
input_sample_rate = data.accelerometer_sample_rate;
output_sample_rate = numel(data.measured_pin_location) / data.measured_pin_location_tt(end);

% compute number of units needed for search space
sample_rates = input_sample_rate ./ downsample_levels;
num_units = zeros([numel(sample_rates),numel(history_lengths)]);
for i = 1:numel(sample_rates)
    for j = 1:numel(history_lengths)
        num_units(i,j) =  (history_lengths(j) * 250 / input_sample_rate) * sample_rates(i);
    end
end
num_units

% set up results grid
[xg,yg]=meshgrid(input_sample_rate./downsample_levels,history_lengths.*250./input_sample_rate);

for downsample = downsample_levels
    for history = history_lengths
        
        % independent variable is the acceleration data
        x = data.acceleration_data';

        % dependent variable is the pin location
        y = data.measured_pin_location';

        % delete nan values from pin position
        nanidx=find(isnan(y));
        for i=nanidx
            legitidx=find(~isnan(y(1:i)));
            last_legitidx=legitidx(end);
            y(i)=y(last_legitidx);
        end
        
        % acceleration data is sampled at a higher rate than pin location data
        
        % downsample X
        x = resample(x,1,downsample);
        
        % upsample Y to match
        [L,M] = rat(numel(x)/numel(y));
        y = resample(y,L,M);
        
        % time shift
        time_shift=0;
        x = x(1+time_shift:end);
        y = y(1:end-time_shift);

        % adjust size
        smallest = int32(min(numel(y),numel(x))*.99);
        x = x(1:smallest);
        y = y(1:smallest);

        % plot all data
        figure;
        subplot(2,1,1);
        plot(x);
        ylabel("accel")
        xlabel("time");
        title("vibration")
        subplot(2,1,2);
        plot(y);
        ylabel("position")
        xlabel("time");
        title("pin position")
        pause(0.01);

        % extract training data
        training_portion = 1;

        training_samples = floor(training_portion*numel(x));
        x_train = x(1:training_samples);
        y_train = y(1:training_samples);

        % check to see if 100% data is for training and extract testing data
        % if needed
        if training_portion == 1
            x_test = [];
            y_test = [];
        else
            x_test = x(training_samples+1:end);
            y_test = y(training_samples+1:end);
        end

        % get mu and sigma for x and y
        mu_x = mean(x_train);
        mu_y = mean(y_train);
        sig_x = std(x_train);
        sig_y = std(y_train);

        % standardize training data
        x_train_standardized = (x_train - mu_x) / sig_x;
        y_train_standardized = (y_train - mu_y) / sig_y;

        % parameters
        numFeatures = 1;
        numResponses = 1;
        
        current_sample_rate = input_sample_rate / downsample;
        history_length = 250 / input_sample_rate * history
        numHiddenUnits = floor(history_length * current_sample_rate)

        % network
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];

        % training parameters
        options = trainingOptions('adam', ...
            'MaxEpochs',100, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',1);

        % train network
        net = trainNetwork(x_train_standardized,y_train_standardized,layers,options);

        % generate predictions for training data
        [net,ypred_standardized] = predictAndUpdateState(net,x_train_standardized);
        ypred = ypred_standardized * sig_y + mu_y;
        rmse = sqrt(mean((ypred-y_train).^2));
        
        %     % plot testing results
        figure
        subplot(3,1,1)
        plot(y_train)
        title("y test")
        xlabel("time")
        ylabel("position")
        subplot(3,1,2)
        plot(ypred)
        ylabel("position")
        title("prediction")
        subplot(3,1,3)
        plot(ypred - y_train)
        xlabel("time")
        ylabel("error")
        title("RMSE = " + rmse)
        
        % log results
        a = downsample/downsample_levels(1);
        b = history/history_lengths(numel(history_lengths));
        temp = whos('net');
        storage(a,b) = temp.bytes/1024;
        throughput(a,b) = 4*numHiddenUnits^2 * current_sample_rate / 1e6;
        rmse_res(a,b) = rmse;
    end
end

figure;
surf(xg,yg,storage);
title('Nominal storage requirements for LSTM structural model');
xlabel('Sample rate (Hz)');
ylabel('History length (s)');
zlabel('Model storage (KiB)');

figure;
surf(xg,yg,throughput);
title('Nominal throughput requirements for LSTM structural model');
xlabel('Sample rate (Hz)');
ylabel('History length (s)');
zlabel('Throughput (Mops/s)');

figure;
surf(xg,yg,rmse_res);
title('Accuracy of LSTM structural model');
xlabel('Sample rate (Hz)');
ylabel('History length (s)');
zlabel('Root mean square error');

% 
% % plot training results
% figure
% subplot(3,1,1)
% plot(y_train)
% title("y train")
% xlabel("time")
% ylabel("position")
% subplot(3,1,2)
% plot(ypred)
% ylabel("position")
% title("prediction")
% subplot(3,1,3)
% plot(ypred - y_train)
% xlabel("time")
% ylabel("error")
% title("RMSE = " + rmse)
% 
% % testing data
% if numel(x_test) ~= 0
%     % generate predictions for testing data
%     % note that net still has state from previous prediction
% 
%     % standardize training data
%     x_test_standardized = (x_test - mu_x) / sig_x;
% 
%     % allocate predicted test data
%     numTimeStepsTest = numel(x_test_standardized);
%     ypred_standardized2 = zeros(1,numTimeStepsTest);
% 
%     for i = 1:numTimeStepsTest
%         [net,ypred_standardized2(:,i)] = predictAndUpdateState(net,x_test_standardized(:,i),'ExecutionEnvironment','cpu');
%     end
% 
%     % un-standardize
%     ypred = sig_y*ypred_standardized2 + mu_y;
% 
%     % caluclate error
%     rmse = sqrt(mean((ypred-y_test).^2))
% 
%     % plot testing results
%     figure
%     subplot(3,1,1)
%     plot(y_test)
%     title("y test")
%     xlabel("time")
%     ylabel("position")
%     subplot(3,1,2)
%     plot(ypred)
%     ylabel("position")
%     title("prediction")
%     subplot(3,1,3)
%     plot(ypred - y_test)
%     xlabel("time")
%     ylabel("error")
%     title("RMSE = " + rmse)
% end
