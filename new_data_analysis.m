% this script is my sandbox for testing low-level and online training

% close plots and clear data from previous run
close all;
profile on

% turn off warnings about curving fitting start points and lack thereof
warning('off','curvefit:fit:noStartPoint');
warning('off','curvefit:fit:nonDoubleYData');

% parameters

% sweep (chose only one)
sweep_precision = 0; % examine impact of fixed point precision
sweep_history = 1; % examine impact of model size
sweep_sample_rate = 0;

% sweep parameters
history_lengths = [1000:100:2000];
precisions = [8];
sample_rates = 1250:1250:5000;

% setting for baseline approach
fft_window = 2048; % in samples
fft_step = 1; % in samples

% ML settings
online_training = 1; % train online or offline (not supported for LSTM)
lstm = 0; % use lstm?  otherwise use mlp
weight_sparsity = .4; % only for LSTM, ignored for MLP
backload_input_samples = 0; % experimental: inteded for use of MLP for Vaheed data; not currently working

% topology, prediction horizon, learning rate
hidden_size = 50; % hidden layer on MLP, ignored for LSTM
prediction_time = 50; % forecast time
alpha = .1; % learning rate
num_lstm_layers = 2; % only for LSTM, ignored for MLPs

% input data preprocessing
subsample_input_signal = 1;

% data settings
use_synthetic_signal = 0;
use_puja_signal = 1;
delete_nonstationarity = 0;
use_vaheed_signal = 0;
nonstationarity_time = 9.775; % only for Puja, ignored for others

% data format
fixed_point = 0; % otherwise use float (not supported for FFT and LSTM: fix this!)

% subsample by changing this to a fixed sample rate
% ignored if subsample_input_signal == 0
model_sample_rate = 20000;

if online_training==1
    epochs = 1;
else
    epochs = 500;
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

% subsample, if needed
if subsample_input_signal==0
    x_sub = x;
    signal_sub = signal;
    model_sample_rate = sample_rate;
    signal_sub_zoh = signal;
else
    sample_period = 1/model_sample_rate;
    subsample = floor(sample_rate / model_sample_rate);
    [x_sub,signal_sub] = myresample(signal,sample_rate,model_sample_rate);
    signal_sub_zoh = myzoh(x,x_sub,signal_sub);
    error_signal = signal - signal_sub_zoh;
    
    % compute subsample SNR
    error_power = rms(error_signal)^2;
    signal_power = rms(signal)^2;
    subsample_snr = log10(signal_power / error_power) * 20
end

perform_fft_forecast (x,x_sub,signal,signal_sub,model_sample_rate,fft_window,fft_step,fft_window,nonstationarity_time);

%return;

% seed RNG
rng(42);

% results
SNR_model = [];
SNR_subsampling = [];
model_snr = [];
subsample_snr = [];
conv_points = [];

if sweep_precision
    sweep_points = numel(precisions);
elseif sweep_history
    sweep_points = numel(history_lengths);
elseif sweep_sample_rate
    sweep_points = numel(sample_rates);
end

conv_times = [];
conv_snrs1 = [];
conv_snrs2 = [];

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
    elseif sweep_sample_rate
        model_sample_rate = sample_rates(i);
        % hold history constant, based on original subsampling rate
        history_length = ceil(history_lengths(1) * sample_rates(i) / sample_rates(1))
        if model_sample_rate == sample_rate
            x_sub = x;
            signal_sub = signal;
        else
            sample_period = 1/model_sample_rate;
            subsample = floor(sample_rate / model_sample_rate);
            [x_sub,signal_sub] = myresample(signal,sample_rate,model_sample_rate);
        end
        
        % compute error signal
        signal_sub_zoh = myzoh(x,x_sub,signal_sub);
        error_signal = signal - signal_sub_zoh;
    end
    
    error_power = rms(error_signal)^2;
    if error_power > 0
        signal_power = rms(signal)^2;
        subsample_snr = [subsample_snr log10(signal_power / error_power) * 20]
    end
    
    % initialize network for low-level trainer
    first_training = 1;

    % allocate training set for offline model
    % if backload is enabled, we predict the first observed input and
    % assume that all the corresponding inputs are 0 (note this only
    % applies to MLP
    if backload_input_samples==0 || lstm==1
        training_samples = numel(signal_sub)-history_length-prediction_time+1;
    else
        training_samples = numel(signal_sub);
    end
    
    if lstm==0
        training_batch_x = zeros(training_samples,history_length);
    end
    
    if lstm==1
        training_samples = numel(signal_sub)-prediction_time;
        training_batch_x = zeros(training_samples,1);
    end
    
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
                if backload_input_samples==0 || lstm==1
                    % normal mode
                    training_batch_x(i,:) =...
                        signal_sub(i:i+history_length-1);

                    training_batch_y(i,1) =...
                        signal_sub(i+history_length+prediction_time-1);
                else
                    % backload mode (for MLP of Vaheed)
                    idx = 1;
                    for j=i-history_length-prediction_time+1:i-prediction_time
                        if j<1
                            training_batch(i,idx) = 0;
                        else
                            training_batch(i,idx) = signal_sub(j);
                        end
                        
                        idx = idx + 1;
                    end
                    
                end
            end
            [mynet,pred,layers] = build_ann (training_batch_x,training_batch_y,[hidden_size],epochs,alpha);
        else
            layers = [ sequenceInputLayer(1) ];
            
            for layer = 1:num_lstm_layers
                layers = [layers lstmLayer(history_length)];
            end

            layers = [layers fullyConnectedLayer(1) regressionLayer];
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
            if weight_sparsity < 1
                net_sparse = sparsify_net(net,weight_sparsity);
            else
                net_sparse = net;
            end
            [net,signal_pred] = predictAndUpdateState(net_sparse,train_x);
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
        % model the error before and after the nonstationarity
        g = fittype('a-b*exp(-c*x)');
        %g = fittype('a*x+b');
        
        % split the time scale
        half_signal_point = find(x>=nonstationarity_time);
        half_signal_point = half_signal_point(1);
        x1 = x(1:half_signal_point);
        x2 = x(half_signal_point+1:end);
        
        % split the error signal
        half1 = error_signal_rms(1:half_signal_point);
        half2 = error_signal_rms(half_signal_point+1:end);
        
        % assume x2 has more points, so trim error
        x2 = x2(1:numel(x1));
        half2 = half2(1:numel(x1));
        
        % fit errors starting at t=0
        model1 = fit(x1',half1',g);
        % start fitting the second error curve at t=0 (use x1)
        %model2 = fit(x1',half2',g,'Lower',[0,-Inf,0],'Upper',[Inf,0,Inf]);
        model2 = fit(x1',half2',g,'Lower',[0,-1e-4,0],'Upper',[Inf,0,1e4]);
        
        if model2.c < 0
            1;
        end
        
        % build error curves starting at t=0
        error_curve1 = model1.a-model1.b*exp(-model1.c.*x1);
        %error_curve1 = model1.a.*x1+model1.b;
        error_curve2 = model2.a-model2.b*exp(-model2.c.*x1);
        %error_curve2 = model2.a.*x1+model2.b;
 
        conv_times = [conv_times -model2.b/model2.c^2]
        
        signal1 = signal_pred_zoh(1:half_signal_point);
        error_signal1 = error_signal(1:half_signal_point);
        conv_snrs1 = [conv_snrs1 log10(rms(signal1)^2/rms(error_signal1)^2)*20]
        
        signal2 = signal_pred_zoh(half_signal_point+1:end);
        error_signal2 = error_signal(half_signal_point+1:end);
        conv_snrs2 = [conv_snrs2 log10(rms(signal2)^2/rms(error_signal2)^2)*20]
        
        model_snr
        
%         conv_point = -log(-.01/(model.b*model.c))/model.c;
%         conv_error = model.a-model.b*exp(-model.c.*conv_point);
%         conv_samples = conv_error * model_sample_rate;
%         conv_points = [conv_points conv_point];
    end
    
    % plot signal and error
    figure;
    subplot (2,1,1);
    plot(x,signal,'b');
    hold on;
    plot(x,signal_pred_zoh,'r');
    %plot(x,error_smooth,'g');
    legend({'signal','predicted signal'});  
    xlabel('time');
    ylabel('accel');
    title("predicted signal, history = "+history_length+" samples");
    
    subplot (2,1,2);
    %plot(x,error_signal,'r');
    plot(x,error_signal_rms,'r');
    hold on;
    if online_training
        %plot(x1,error_curve1,'g');
        plot(x2,error_curve2,'g');
    	%legend({'error signal rms','error curve1','error curve2'});
        legend({'error signal rms','error fit'});
    else
        legend({'error signal rms'});
    end
    
    xlabel('time');
    ylabel('accel');
    title("predicted signal, history = "+history_length);
    hold off;
    drawnow;
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
elseif sweep_sample_rate
    x_vals = sample_rates;
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
    
    if online_training
        figure;
        plot(x_vals ./ model_sample_rate .* 1000,conv_times .* 1000);
        title('Convergence time after nonstationarity');
        xlabel('model history length (ms)');
        ylabel('convergence time (ms)');
    end
    
    if online_training
        figure
        plot(x_vals ./ model_sample_rate .* 1000,conv_snrs2);
        title('SNR after converging on nonstationarity');
        xlabel('model history length (ms)');
        ylabel('SNR (dB)');
    end
    
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
    
elseif sweep_sample_rate

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
    xlabel('model sample rate');
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

