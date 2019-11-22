close all;
clear all;

if ~isfile('data_6_with_FFT.json')
    fprintf('unzipping datafile...\n');
    system('7za e data_6_with_FFT.7z');
end

% measure the storage, throughput, and accuracy given by different
% frequencies and history length
downsample_levels = 8:8:64; % relative to input sample rate
history_multiple = 1000;      % relative to input sample rate
history_lengths = 5:-1:1;

% allocate measurement data
storage = zeros(numel(downsample_levels),numel(history_lengths));
throughput = zeros(numel(downsample_levels),numel(history_lengths));
rmse_res = zeros(numel(downsample_levels),numel(history_lengths));

% read data
data = jsondecode(fileread('data_6_with_FFT.json'));
input_sample_rate = data.accelerometer_sample_rate;
output_sample_rate = numel(data.measured_pin_location) / data.measured_pin_location_tt(end);

% set up results grid
[xg,yg]=meshgrid(input_sample_rate./downsample_levels,history_lengths.*history_multiple./input_sample_rate);

% counters for recording results
dd=1;
hh=1;

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

%         % plot all data
%         figure;
%         subplot(2,1,1);
%         plot(x);
%         ylabel("accel")
%         xlabel("time");
%         title("vibration")
%         subplot(2,1,2);
%         plot(y);
%         ylabel("position")
%         xlabel("time");
%         title("pin position")
%         pause(0.01);

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
        history_length = history_multiple / input_sample_rate * history
        numHiddenUnits = floor(history_length * current_sample_rate)

        % network
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];

        % training parameters
        options = trainingOptions('adam', ...
            'MaxEpochs',200, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.0005, ...
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
        
        % plot testing results
        %figure
        %subplot(3,1,1)
        %time=resample(data.measured_pin_location_tt,L,M);
        
        %time_span = data.measured_pin_location_tt(end) - data.measured_pin_location_tt(1);
        %time_increment = time_span / numel(y_train);
        %time=[data.measured_pin_location_tt(1) : time_increment : data.measured_pin_location_tt(end)];
        %time=time(1:numel(y_train));
        %time=linspace(data.measured_pin_location_tt(1),data.measured_pin_location_tt(end),numel(y_train));
        
        %temp = whos('net');
        
        %subplot(3,1,1);
        %plot(time,y_train,'b');
        %xlabel("time (s)");
        %ylabel("actual pin position (m)");
        
        %subplot(3,1,2);
        %plot(time,ypred,'r');
        %xlabel("time (s)");
        %ylabel("predicted pin position (m)");
        
        %subplot(3,1,3);
        %hold on;
        %plot(time,y_train,'b');
        %plot(time,ypred,'r');
        %xlabel("time (s)");
        %ylabel("actual+predicted pin position (m)");
        
        %subplot(3,1,1);
        %title("Actual/predicted pin position, " +...
        %            current_sample_rate +...
        %            "Hz, " +...
        %            numHiddenUnits +...
        %            " units, " +...
        %            4*numHiddenUnits^2 * current_sample_rate / 1e6 +...
        %            " sustained Mops, " +...
        %            temp.bytes/1024 +...
        %            "KiB uncomp storage, " +...
        %            history_length*1000 +...                        
        %            "ms history " +...
        %            " RMSE = " +...
        %            rmse);
        %hold off;
        %pause(0.01);
          
        % plot testing results
        %figure
        %subplot(3,1,1)
       
        %temp = whos('net');
        
        %range1 = [1:numel(y_train)];
        %ax1=subplot(2,1,1);
        
        %t1=time(range1);
        %xts1=x_train_standardized(range1);
        
        %plot(t1,xts1,'b');
        %xlabel("time (s)");
        %ylabel("input data (accel)");
        
        %ax2=subplot(2,1,2);
        
        %yp1=ypred(range1);
        
        %plot(t1,yp1,'r');
        %xlabel("time (s)");
        %ylabel("predicted pin position (m)");
        
        %linkaxes([ax1 ax2],'x');
        
        %ylabel("position")
        %subplot(3,1,2)
        %ylabel("position")
%         title("prediction")
%         subplot(3,1,3)
%         plot(ypred - y_train)
%         xlabel("time")
%         ylabel("error")
        %title("RMSE = " + rmse)
        
        % log results
        a = dd;
        b = hh;
        temp = whos('net');
        storage(a,b) = temp.bytes/1024;
        throughput(a,b) = 4*numHiddenUnits^2 * current_sample_rate / 1e6;
        rmse_res(a,b) = rmse;
        hh=hh+1;
    end
    dd=dd+1;
    hh=1;
end

figure;
surf(xg,yg,storage');
title('Nominal storage requirements for LSTM structural model');
xlabel('Sample rate (Hz)');
ylabel('History length (s)');
zlabel('Model storage (KiB)');
colorbar;

figure;
surf(xg,yg,throughput');
title('Nominal throughput requirements for LSTM structural model');
xlabel('Sample rate (Hz)');
ylabel('History length (s)');
zlabel('Throughput (Mops/s)');
colorbar;

figure;
surf(xg,yg,rmse_res');
title('Accuracy of LSTM structural model');
xlabel('Sample rate (Hz)');
ylabel('History length (s)');
zlabel('Root mean square error');
colorbar;

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
