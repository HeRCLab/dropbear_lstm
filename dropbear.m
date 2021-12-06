clc;
close all;
clear all;

% measure the storage, throughput, and accuracy given by different
% frequencies and history length
downsample_levels = 56:8:72; % relative to input sample rate
downsample = 64;
%history_lengths = 10:-1:1;     % relative to 25 samples at input sample rate
history_lengths = [10:10:201];
% LearnRateDropPeriods = [50:50:251];
% LearnRateDropFactors = [0.1:0.1:0.51];
% MiniBatchSizes = [8, 16, 32, 64, 128, 256, 512];

% allocate measurement data
storage = zeros(numel(downsample_levels),numel(history_lengths));
throughput = zeros(numel(downsample_levels),numel(history_lengths));
rmse_res = zeros(numel(downsample_levels),numel(history_lengths));

% read data
data = jsondecode(fileread('\Users\bpriddy\Documents\GitHub\dropbear_lstm\data_6_with_FFT.json'));
input_sample_rate = data.accelerometer_sample_rate;
output_sample_rate = numel(data.measured_pin_location) / data.measured_pin_location_tt(end);

% compute number of units needed for search space
sample_rates = input_sample_rate ./ downsample_levels;
num_units = zeros([numel(sample_rates),numel(history_lengths)]);

%saving rmse/error values to json file

[fid, msg] = fopen('\Users\bpriddy\Downloads\LSTM\lstm_gridsearch_stacked3.json', 'w');
s = struct;

for i = 1:numel(sample_rates)
    for j = 1:numel(history_lengths)
        num_units(i,j) =  (history_lengths(j) * 250 / input_sample_rate) * sample_rates(i);
    end
end
num_units;

% set up results grid
[xg,yg]=meshgrid(input_sample_rate./downsample_levels,history_lengths.*250./input_sample_rate);



idx = 1;


    for history = history_lengths

          
          
          
%         [fid, msg] = fopen('\Users\bpriddy\Downloads\LSTM\lstm_dsf_time2.json', 'a+');
        
        model_snr = [];
        subsample_snr = [];
        % independent variable is the acceleration data
        x = data.acceleration_data';

        % dependent variable is the pin location
        y = data.measured_pin_location';
        signal = y;
        % time value
        time = data.time_acceleration_data';
        [L,M] = rat(numel(x)/numel(y));
        time = resample(time,M,L);
       
       
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
        time_sub = resample(time,L,M);
       
        % time shift
        time_shift=0;
        x = x(1+time_shift:end);
        y = y(1:end-time_shift);

        % adjust size
        smallest = int32(min(numel(y),numel(x))*.99);
        x = x(1:smallest);
        y = y(1:smallest);
        time_sub = time_sub(1:smallest);

        signal_zoh = myzoh(time,time_sub,double(y));
        % compute error signal
        error_signal = signal - signal_zoh;
        % snip lead-in for MLP
       
        error_signal_snip = error_signal;
        signal_snip = signal;
       
   
        % compute error power
%         error_power = rms(error_signal_snip)^2;
        % compute signal power
%         signal_power = rms(signal_snip)^2;
        rmean = [];
        emean = [];
        for i=1:(numel(signal_snip))
            if isnan(signal_snip(i)) == 1
                rmean = [rmean, (signal_snip(i+1))^2];
            else
                rmean = [rmean, (signal_snip(i))^2];
            end
           
           
            if isnan(error_signal_snip(i)) == 1
                emean = [emean, (error_signal_snip(i+1))^2];
            else
                emean = [emean, (error_signal_snip(i))^2];
            end
           
        end
        %the actual signal and error power are just Vrms, but this is
        %accounted for when calculating the SNR
        
        signal_power = sqrt(sum(rmean)/numel(signal_snip));
        error_power = sqrt(sum(emean)/numel(error_signal_snip));
        % compute SNR
%         model_snr = [model_snr,log10(signal_power / error_power) * 20];
        % comptue instantaneous RMS
 
       
         %Plot "Zero Order Hold vs Actual"
%         figure;
%         plot(time, signal_zoh, 'r-o', time, signal, 'b');
%         ylabel("accel");
%         xlabel("time");
%         title("Zero Order Hold vs Actual");
%         legend("ZOH", "Actual");
       
       
        % plot all data
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
        numHiddenUnits = history;
       
        
        
        current_sample_rate = input_sample_rate / downsample;
        history_length = 250 / input_sample_rate * history;
        %numHiddenUnits = floor(history_length * current_sample_rate)
        
% 
%         name = sprintf("ModelSNR_DSF%d_HiddenUnits%d", downsample, numHiddenUnits);
%         s.(name) = model_snr;
        
%         lgraph = layerGraph();
%         
%         tempLayers = sequenceInputLayer(1,"Name","sequence");
%         lgraph = addLayers(lgraph,tempLayers);
% 
%         tempLayers = lstmLayer(numHiddenUnits2,"Name","lstm_1");
%         lgraph = addLayers(lgraph,tempLayers);
% 
%         tempLayers = lstmLayer(numHiddenUnits,"Name","lstm_2");
%         lgraph = addLayers(lgraph,tempLayers);
% 
%         tempLayers = [
%             concatenationLayer(1,2,"Name","concat")
%             lstmLayer((numHiddenUnits+numHiddenUnits2),"Name","lstm_3")
%             fullyConnectedLayer(1,"Name","fc")
%             regressionLayer("Name","regressionoutput")];
%             lgraph = addLayers(lgraph,tempLayers);
% 
%             % clean up helper variable
%         clear tempLayers;
% 
%         lgraph = connectLayers(lgraph,"sequence","lstm_1");
%         lgraph = connectLayers(lgraph,"sequence","lstm_2");
%         lgraph = connectLayers(lgraph,"lstm_1","concat/in1");
%         lgraph = connectLayers(lgraph,"lstm_2","concat/in2");
%         
        
        % network
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(numHiddenUnits)
            lstmLayer(numHiddenUnits)
            lstmLayer(numHiddenUnits)
            lstmLayer(numHiddenUnits)
            fullyConnectedLayer(numResponses)
            regressionLayer];

        
        % training parameters
        options = trainingOptions('adam', ...
            'MaxEpochs',1000, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'MiniBatchSize',128, ...
            'Shuffle','never', ...
            'Verbose',0, ...
            'Plots', 'training-progress', ...
            'OutputFcn',@(info)stopIfAccuracyNotImproving(info,5));

%         display(LearnRateDropPeriod);
%         display(LearnRateDropFactor);
%         display(MiniBatchSize);
        
%         name = sprintf("LearnRateDropPeriod_LRDP%d_LRDF%d_batch%d", LearnRateDropPeriod, int8(LearnRateDropFactor*100), MiniBatchSize);
%         s.(name) = LearnRateDropPeriod;
%         
%         name = sprintf("LearnRateDropFactor_LRDP%d_LRDF%d_batch%d", LearnRateDropPeriod, int8(LearnRateDropFactor*100), MiniBatchSize);
%         s.(name) = LearnRateDropFactor;
%         
%         name = sprintf("MiniBatchSize_LRDP%d_LRDF%d_batch%d", LearnRateDropPeriod, int8(LearnRateDropFactor*100), MiniBatchSize);
%         s.(name) = MiniBatchSize;
        % train network
       
        display(numHiddenUnits);
        [net, info] = trainNetwork(x_train_standardized,y_train_standardized,layers,options);
        
       
        % generate predictions for training data
        [net,ypred_standardized] = predictAndUpdateState(net,x_train_standardized);
        ypred = ypred_standardized * sig_y + mu_y;
        rmse = sqrt(mean((ypred-y_train).^2));
       
        
       
       
        signal_pred_zoh = myzoh(time,time_sub,double(ypred));
        % compute error signal
        error_signal = signal - signal_pred_zoh;
        % snip lead-in for MLP
       
        error_signal_snip = error_signal;
        signal_snip = signal;
       
   
        % compute error power
%         error_power = rms(error_signal_snip)^2;
        % compute signal power
%         signal_power = rms(signal_snip)^2;
        rmean = [];
        emean = [];
        for i=1:(numel(signal_snip))
            if isnan(signal_snip(i)) == 1
                rmean = [rmean, (signal_snip(i+1))^2];
            else
                rmean = [rmean, (signal_snip(i))^2];
            end
           
           
            if isnan(error_signal_snip(i)) == 1
                emean = [emean, (error_signal_snip(i+1))^2];
            else
                emean = [emean, (error_signal_snip(i))^2];
            end
           
        end    
        signal_power = sqrt(sum(rmean)/numel(signal_snip));
        error_power = sqrt(sum(emean)/numel(error_signal_snip));


        % compute SNR
        subsample_snr = [subsample_snr,log10(signal_power / error_power) * 20];
        % comptue instantaneous RMS
       
        
        name = sprintf("SNR4_Units%d", numHiddenUnits);
        s.(name) = subsample_snr;
        
        name = sprintf("RMSE4_Units%d", numHiddenUnits);
        s.(name) = rmse;
        
%         delete(findall(0));  %close all figs
        %name = sprintf("SubsampleSNR_DSF%d_HiddenUnits%d", downsample, numHiddenUnits);
        %class(name)
            % plot testing results
        figure;
        subplot(3,1,1);
        plot(y_train);
        title("y test");
        xlabel("time");
        ylabel("position");
        subplot(3,1,2);
        plot(ypred);
        ylabel("position");
        title("prediction");
        subplot(3,1,3);
        plot(ypred - y_train);
        xlabel("time");
        ylabel("error");
        title("SNR = " + subsample_snr);
        saveas(gcf, name, 'fig');
       
%         if idx < 5
%             idx = idx + 1
%         end
        % log results
%         a = downsample/downsample_levels(1);
%         b = history/history_lengths(numel(history_lengths));
%         temp = whos('net');
%         storage(a,b) = temp.bytes/1024;
%         throughput(a,b) = 4*numHiddenUnits^2 * current_sample_rate / 1e6;
%         rmse_res(a,b) = rmse;
    end


jsonText2 = jsonencode(s);
fprintf(fid, jsonText2);
fclose(fid);

%
% figure;
% surf(xg,yg,storage');
% title('Nominal storage requirements for LSTM structural model');
% xlabel('Sample rate (Hz)');
% ylabel('History length (s)');
% zlabel('Model storage (KiB)');
%
% figure;
% surf(xg,yg,throughput');
% title('Nominal throughput requirements for LSTM structural model');
% xlabel('Sample rate (Hz)');
% ylabel('History length (s)');
% zlabel('Throughput (Mops/s)');
%
% figure;
% surf(xg,yg,rmse_res');
% title('Accuracy of LSTM structural model');
% xlabel('Sample rate (Hz)');
% ylabel('History length (s)');
% zlabel('Root mean square error');
% diary off

function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end

function signal = myzoh (x,x_sub,signal_in)
   
    signal = zeros(1,size(x,2));
   
    i_sub = 1;
       
    for i=1:size(x,2)
        signal(1,i) = signal_in(1,i_sub);
       
        while (x(1,i) >= x_sub(1,i_sub)) && (i_sub<size(signal_in,2))
            i_sub = i_sub+1;
        end
       
       
    end

end
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

