% this script is my sandbox for testing low-level and online training

% close plots and clear data from previous run
close all;
clear all;
profile on

% parameters
sweep_precision = 0; % examine impact of fixed point precision
sweep_history = 1; % examine impact of model size
online_training = 0; % train online or offline
hidden_size = 10; % hidden layer on MLP, ignored for LSTM
prediction_time = 20; % forecast time
alpha = 1e-1; % learning rate
lstm = 0; % use lstm?  otherwise use mlp

% data settings
use_synthetic_signal = 0;
use_puja_signal = 1;
delete_nonstationarity = 1;
use_vaheed_signal = 0;

% data format
fixed_point = 0; % otherwise use float

if online_training==1
    epochs = 1;
else
    epochs = 10;
end

if use_synthetic_signal
    % % signal synthesis
    sample_rate = 50000;
    time = 0.5;
    amps=[1, 1, 1];
    freqs=[100, 150, 400];
    phases=[0, 0, 0];
    % synthesize signal
    [x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

elseif use_puja_signal
    % read Puja's signal
    %data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
    %data=lvm_import('Ivol_Acc_Load_data1_w3_NSTD.lvm');
    data=lvm_import('data_set_3.lvm');
    x = data.Segment1.data(:,1)';
    signal = data.Segment1.data(:,4)';
    if delete_nonstationarity
        subset = floor(numel(signal)/2.2);
        signal = signal(1:subset);
        x = x(1:subset);
    end
    
elseif use_vaheed_signal
    % Drop Tower data
    droptowerdata = importfile_droptower("drop_tower_data.csv");
    x = droptowerdata.time';
    signal = droptowerdata.test1';
end

% set up the time axis
time_span = x(end) - x(1);
sample_rate = numel(x)/time_span;
time_offset = x(1);

% subsample by changing this to a fixed sample rate (not equalling
% native sample rate)
model_sample_rate = 1250;

% subsample, if needed
if model_sample_rate == sample_rate
    x_sub = x;
    signal_sub = signal;
else
    sample_period = 1/model_sample_rate;
    subsample = floor(sample_rate / model_sample_rate);
    [x_sub,signal_sub] = myresample(signal,sample_rate,model_sample_rate);
end

% seed RNG
rng(42);

SNR_model = [];
SNR_subsampling = [];

% % plot original signal
% figure;
% plot(x,signal,'k');
% hold on;
% xlabel('time');
% legend({'original signal'});
% xlim ([4.35 4.38]);
% hold off;

% plot subsampled signal
% figure;
% plot(x,signal,'k');
% hold on;
% plot(x,signal_sub_zoh,'r');
% %plot(x,error_signal,'r');
% %legend({'original signal','subsampled signal','error signal'});
% legend({'original signal','subsampled signal'});
% xlabel('time');
% xlim ([4.35 4.38]);
% hold off;
% drawnow;

history_lengths = 100:50:300;
precisions = [5:12];
history_length = 50;

model_snr = [];
subsample_snr = [];
conv_points = [];

if sweep_precision
    sweep_points = numel(precisions);
elseif sweep_history
    sweep_points = numel(history_lengths);
end

% outermost loop:  generate models for each sweep point
for i=1:sweep_points

    if sweep_precision
        wordsize = precisions(i);
        fractionsize = wordsize-1;
        % determine SNR of subsampling/quantization
        % convert subsampled signal back to the time domain of the original signal
        signal_sub_zoh = double(myzoh(x,x_sub,fi(signal_sub,1,wordsize,fractionsize)));

        % compute error signal
        error_signal = signal - signal_sub_zoh;
        history_length = history_lengths(1);
    elseif sweep_history
        wordsize = precisions(1);
        fractionsize = wordsize-1;
        
        % compute error signal
        signal_sub_zoh = myzoh(x,x_sub,signal_sub);
        error_signal = signal - signal_sub_zoh;
        history_length = history_lengths(i);
    end
    
    error_power = rms(error_signal)^2;
    if error_power > 0
        signal_power = rms(signal)^2;
        subsample_snr = [subsample_snr log10(signal_power / error_power) * 20];    
    end
    
    % initialize network for low-level trainer
    first_training = 1;

    % allocate training set for offline model
    training_samples = numel(signal_sub)-history_length-prediction_time+1;
    training_batch_x = zeros(training_samples,history_length);
    training_batch_y = zeros(training_samples,1);

    % allocate and zero-pad predicted signal
    signal_pred = zeros(1,training_samples);

    % type cast predicted signal as fixed point if necessary
    if fixed_point==1
        signal_pred = fi(signal_pred,1,wordsize,fractionsize);
    end

    % OFFLINE TRAINING SECTION
    if online_training==0
        if lstm==0
            % build training set using format acceptable for Matlab-based
            % trainer
            for i = 1:training_samples
                training_batch_x(i,:) =...
                    signal_sub(i:i+history_length-1);

                training_batch_y(i,1) =...
                    signal_sub(i+history_length+prediction_time-1);
            end
            [mynet,pred,layers] = build_ann (training_batch_x,training_batch_y,[hidden_size],epochs,alpha);
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
            net = trainNetwork(train_x,training_batch_y,layers,opts);
        else
            train_x = signal_sub(1:end-prediction_time);
            train_y = signal_sub(prediction_time+1:end);
            net = trainNetwork(train_x,train_y,layers,opts);
        end
        
        % predict
        if fixed_point==1
           mynet.weights_hidden = fi(net.Layers(2,1).Weights,1,wordsize,fractionsize,'OverflowAction','Wrap');
           mynet.bias_hidden = fi(net.Layers(2,1).Bias,1,wordsize,fractionsize,'OverflowAction','Wrap');
           mynet.weights_output = fi(net.Layers(3,1).Weights,1,wordsize,fractionsize,'OverflowAction','Wrap');
           mynet.bias_output = fi(net.Layers(3,1).Bias,1,wordsize,fractionsize,'OverflowAction','Wrap');
        end

        if lstm == 1
            [net,signal_pred] = predictAndUpdateState(net,train_x);
            % phase shift it!
            fill = zeros(1,prediction_time);
            signal_pred = [fill signal_pred(1,1:end-prediction_time-1)];
            
        else
            % MLP
            input_val = zeros(history_length,1,1);
            for i=1:training_samples
                input_val(:,1,1) = signal_sub(i:i+history_length-1);
                if fixed_point == 0
                    signal_pred(i+history_length+prediction_time-1) = predict(net,input_val);
                else
                    input_val = fi(input_val,1,wordsize,fractionsize);
                    out_val = mypredict2(mynet,input_val);
                    signal_pred(i+history_length+prediction_time-1) = out_val;
                end
            end

        end
    end

    first_training = 1;
    
    % ONLINE TRAINING SECTION
    if online_training==1

         for i = 1:training_samples

            % train one sample at a time
            x_train = signal_sub(i:i+history_length-1);
            y_train = signal_sub(i+history_length+prediction_time-1);

            % train network
            if first_training
                [mynet,output_from_mlp] = build_ann(x_train,y_train,[hidden_size],epochs,alpha);
                first_training=0;
            else
                [mynet,output_from_mlp] = build_ann(x_train,y_train,[hidden_size],epochs,alpha,mynet);
            end

            %x_pred = signal_sub(i-history_length+1:i);
            prediction = mypredict(mynet,x_train);
            index = i+history_length+prediction_time-1;
            signal_pred(1,index)=prediction;

%             % handle all the plotting
%             if mod(i,1000)==0
%                 % plot
%                 time_axis = 0:(size(signal_pred,2)-1);
%                 time_axis = time_axis ./ model_sample_rate + time_offset;
% 
%                 % plot the actual and predicted signal
%                 subplot(2,1,1);
%                 plot(time_axis,signal_sub(1,1:size(signal_pred,2)),'b');
%                 hold on;
%                 plot(time_axis,signal_pred,'r');
%                 legend ('signal (actual)','signal (predicted)','Location','southeast');
% 
%                 subplot(2,1,2);
%                 % plot the cumulative RMS error
%                 rmse = zeros(1,size(signal_pred,2));
%                 for i=1:size(signal_pred,2)
%                     rmse(1,i) = mean((double(signal_pred(1,1:i)) - signal_sub(1,1:i)).^2)^.5;
%                 end
%                 
%                 plot(time_axis,rmse(1,1:size(time_axis,2)),'r');
%                 legend ({'cumulative RMS error'},'Location','southeast');
%                 title('cumulative RMS error');
%                 %rmse(1,end)
%                 drawnow;
%             end
          end
         

    end
    
    % analyze the signal
    
    % determine SNR of the model
    % convert subsampled signal back to the time domain of the original signal
    signal_pred_zoh = myzoh(x,x_sub,double(signal_pred));
    % compute error signal
    error_signal = signal - signal_pred_zoh;
    % snip lead-in for MLP
    if ~lstm
        error_signal_snip = error_signal(history_length+prediction_time-1:end);
        signal_snip = signal(history_length+prediction_time-1:end);
    else
        error_signal_snip = error_signal;
        signal_snip = signal;
    end
    
    % compute error power
    error_power = rms(error_signal_snip)^2;
    % compute signal power
    signal_power = rms(signal_snip)^2;
    % compute SNR
    model_snr = [model_snr,log10(signal_power / error_power) * 20]
    % comptue instantaneous RMS
    error_signal_rms = (error_signal .^ 2) .^ .5;
    
    if online_training
        % find the nonstationarity event
        filter_width = 10;
        
        error_smooth = filter(ones(1,filter_width),...
                              ones(1,filter_width)*filter_width,...
                              error_signal_rms);                          
        
        
        g = fittype('a-b*exp(-c*x)');
        
        half_signal_point = floor(size(error_signal_rms,2)/2);
        x1 = x(1:half_signal_point);
        x2 = x(half_signal_point+1:end);
        half1 = error_signal_rms(1:half_signal_point);
        half2 = error_signal_rms(half_signal_point+1:end);
        
        model1 = fit(x1',half1',g);
        model2 = fit(x2',half2',g);
        
        error_curve1 = model1.a-model1.b*exp(-model1.c.*x1);
        error_curve2 = model2.a-model2.b*exp(-model2.c.*x2);
        
%         conv_point = -log(-.01/(model.b*model.c))/model.c;
%         conv_error = model.a-model.b*exp(-model.c.*conv_point);
%         conv_samples = conv_error * model_sample_rate;
%         conv_points = [conv_points conv_point];
    end
    
    figure;
    subplot (2,1,1);
    plot(x,signal,'b');
    hold on;
    plot(x,signal_pred_zoh,'r');
    %plot(x,error_smooth,'g');
    legend({'signal','predicted signal'});  
    xlabel('time');
    ylabel('accel');
    title("predicted signal, history = "+history_length);
    
    subplot (2,1,2);
    %plot(x,error_signal,'r');
    plot(x,error_signal_rms,'r');
    hold on;
    if online_training
        plot(x1,error_curve1,'g');
        plot(x2,error_curve2,'g');
    	legend({'error signal rms','error curve1','error curve2'});
    else
        legend({'error signal rms'});
    end
    
    xlabel('time');
    ylabel('accel');
    title("predicted signal, history = "+history_length);
    hold off;
end
    
% float
deploy.DSP = [7,15,32,30,46,64,41,60,65,92];
deploy.FF = [2,4,9,8,12,18,12,17,18,25];
deploy.LUT = [4,8,15,16,23,31,24,32,34,45];
deploy.latency = [345,379,379,421,421,421,478,455,455,455];

% precision
deploy.DSP = [0,0,0,0,0,0,0];
deploy.FF = [1,2,2,2,2,2,2];
deploy.LUT = [33,33,34,37,40,43,46];
deploy.latency = [38,46,51,59,60,60,53];

% precision
deploy.DSP = [0,0,29,29];
deploy.FF = [4,4,5,5];
deploy.LUT = [59,69,61,69];
deploy.latency = [54,60,68,65];

% precision
deploy.DSP = [];
deploy.FF = [];
deploy.LUT = [];
deploy.latency = [];

% plot SNR results

%x_vals = history_lengths./model_sample_rate;
if sweep_precision
    x_vals = precisions;
elseif sweep_history
    x_vals = history_lengths;
end

figure;

% plot subsampling SNR
% only plot subsampling error if subsampling was performed
if ~isempty(subsample_snr)
    plot(x_vals,subsample_snr,'r--');
end

hold on;
% plot model SNR
plot(x_vals,model_snr,'go-');
%ylim([0 18]);
%xlim([0.0025 0.045]);

if sweep_history
    
    % print costs if available
    if ~isempty(deploy.DSP)
        for i=1:numel(history_lengths)
            gstr = sprintf("%d%% DSP\n%d%% LUT\n%d%% FF,\nlatency %d ns",deploy.DSP(i),...
                                                                deploy.LUT(i),...
                                                                deploy.FF(i),...
                                                                deploy.latency(i));
            %plot(x_vals(i),model_snr(i),'r.');
            text(x_vals(i),model_snr(i)-1,gstr);
        end
    end
    
    if ~isempty(subsample_snr)
        legend({'subsampling SNR','model SNR'});
    else
        legend({'model SNR'});
    end

    if ~lstm
        xlabel('model history length');
    else
        xlabel('number of units');
    end
    ylabel('SNR');
    
elseif sweep_precision

    % print costs if available
    if ~isempty(deploy.DSP)
        for i=1:numel(precisions)
            gstr = sprintf("%d%% DSP\n%d%% LUT\n%d%% FF,\nlatency %d ns",deploy.DSP(i),...
                                                                deploy.LUT(i),...
                                                                deploy.FF(i),...
                                                                deploy.latency(i));
            %plot(x_vals(i),model_snr(i),'r.');
            text(x_vals(i),model_snr(i)-2,gstr);
        end
    end
    
    if ~isempty(subsample_snr)
        legend({'subsampling SNR','model SNR'});
    else
        legend({'model SNR'});
    end
    
    %xlabel('model history length');
    xlabel('fixed point precision');
    ylabel('SNR');
    xlim([-1,11]);
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
