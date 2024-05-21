% this script is my sandbox for testing low-level and online training

% close plots and clear data from previous run
close all;
clear all;

% turn off warnings about curving fitting start points and lack thereof
warning('off','curvefit:fit:noStartPoint');
warning('off','curvefit:fit:nonDoubleYData');

% enable plotting
plotit=1;

% for plots
fontsize = 20;

% sweep (chose only one)
sweep_precision = 0; % examine impact of fixed point precision
sweep_history = 0; % examine impact of model size
sweep_sample_rate = 1; % examine impact of history size

% sweep parameters
history_lengths = [40e-3];
precisions = [4];
sample_rates = 2500:2500:20000;

% subsample by changing this to a fixed sample rate
% ignored if subsample_input_signal == 0
model_sample_rate = 12500;

% setting for baseline approach
fft_window_seconds = 1e-1; % in seconds; the period of the signal
fft_step = 1; % in samples

% ML settings
online_training = 1; % train online or offline (not supported for LSTM)
lstm = 0; % use lstm?  otherwise use mlp

% topology, prediction horizon, learning rate
hidden_size = 50; % hidden layer on MLP, ignored for LSTM
prediction_time_seconds = 40e-3; % forecast time in seconds
alpha = .1; % learning rate
num_lstm_layers = 2; % only for LSTM, ignored for MLPs

% input data preprocessing
subsample_input_signal = 1;

% data settings
use_synthetic_signal = 0;
use_puja_signal = 1;
use_vaheed_signal = 0;

delete_nonstationarity = 0;
nonstationarity_time = 9.775; % only for Puja, ignored for others

% data format
fixed_point = 0; % otherwise use float (not supported for FFT and LSTM: fix this!)

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
    %data=lvm_import('new_data_sets/Test 1.lvm');
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
    
    % optionally plot the subsampling SNR
    if plotit
        fontsize = 14;
        figure;
        plot(x,signal,'b');
        xlim([9 9.05]);
        hold on;
        plot(x,signal_sub_zoh,'r');
        %plot(x,error_smooth,'g');
        legend({'signal','subsampled'},'interpreter','latex');
        xlabel('$t$','interpreter','latex');
        %ylabel('acceleration','interpreter','latex');
        title("$r_s$ = "+model_sample_rate,'interpreter','latex');
        set(gca,'FontSize',fontsize);
        set(gca,'TickLabelInterpreter','latex')
    end
end

% compute the prediction (forecase time) in samples
prediction_time = ceil(prediction_time_seconds * sample_rate);

% compute the FFT window (forecase time) in samples
fft_window = ceil(fft_window_seconds * model_sample_rate);
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
snr_before_nonstationarity = [];
a_vals = [];
b_vals = [];
c_vals = [];

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
        history_length = ceil(history_lengths(1) * model_sample_rate);
    elseif sweep_history
        wordsize = precisions(1);
        fractionsize = wordsize-1;
        
        % compute error signal
        signal_sub_zoh = myzoh(x,x_sub,signal_sub);
        error_signal = signal - signal_sub_zoh;
        history_length = ceil(history_lengths(i) * model_sample_rate);
    elseif sweep_sample_rate
        model_sample_rate = sample_rates(i);
        % hold history constant, based on original subsampling rate
        history_length = ceil(history_lengths(1) * model_sample_rate)
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

    training_samples = numel(signal_sub)-history_length-prediction_time+1;
    
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
        x_snip = x(history_length+prediction_time-1:end);
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
    
    if plotit
        figure;
        plot(x(1:end-prediction_time),signal(1:end-prediction_time),'r-');
        hold on;
        plot(x(1:end-prediction_time),signal_pred_zoh(prediction_time+1:end),'b-');
        legend({'$V(t)$','$V_{forecast}(t-f/r_s)$'},'interpreter','latex');
        title('$s$=50, $f/r_s$=40 ms','interpreter','latex');
        xlabel('time','interpreter','latex');
        %ylabel('$SNR_{db}$','interpreter','latex');
        set(gca,'FontSize',fontsize);
        set(gca,'TickLabelInterpreter','latex');
        %ax = ancestor(h, 'axes');
        %ax.XAxis.Exponent = 0;
        hold off;
    end
    
    % SNR before non-stationarity
    ns_sample = find(x_snip<nonstationarity_time);
    ns_sample = ns_sample(end);
    error_signal_before_nonstationarity = error_signal_snip(1:ns_sample);
    signal_before_nonstationarity = signal_snip(1:ns_sample);
    snr_before_nonstationarity = [snr_before_nonstationarity, log10(rms(signal_before_nonstationarity)^2 /...
                                       rms(error_signal_before_nonstationarity)^2) * 20]
    if online_training
        [snr,conv_time,a,b,c] = get_accuracy_stats (x,signal,signal_pred_zoh,error_signal_rms,nonstationarity_time,plotit,history_length,"MLP-Based Model with $h$="+history_length);
        a_vals = [a_vals a];
        b_vals = [b_vals b];
        c_vals = [c_vals c];
        conv_times = [conv_times conv_time]
        %conv_snrs1 = [conv_snrs1 log10(rms(signal1)^2/rms(error_signal1)^2)*20]
        conv_snrs2 = [conv_snrs2 snr]
        model_snr
    end
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

% find the x-axis (depends on what parameter is being swept)
if sweep_precision
    x_vals = precisions;
elseif sweep_history
    %x_vals = history_lengths ./ model_sample_rate .* 1000;
    x_vals = history_lengths .* 1000;
elseif sweep_sample_rate
    x_vals = sample_rates;
end

% FIGURE 1: overall SNR
if ~exist('snr_fig','var')
    snr_fig=figure;
    % plot subsampling SNR
    % only plot subsampling error if subsampling was performed
    if ~isempty(subsample_snr)
        plot(x_vals,subsample_snr,'r--','DisplayName','subsampling SNR');
    end
else
    figure(snr_fig);
end
hold on;

if ~fixed_point
        display_name = hidden_size+" hidden neurons";
        if hidden_size==10
           plotopts = 'rs-';
        elseif hidden_size==20
           plotopts = 'gd-';
        elseif hidden_size==30
            plotopts = 'b^-';
        elseif hidden_size==40
            plotopts = 'mv-';
        else
            plotopts = 'cx-';
        end
    else
        display_name = wordsize+" bits";
        if wordsize==4
           plotopts = 'rs-';
        elseif wordsize==5
            plotopts = 'gd-';
        elseif wordsize==6
            plotopts = 'b^-';
        elseif wordsize==7
            plotopts = 'mv-';
        else
            plotopts = 'cx-';
        end
    end

% plot model SNR
plot(x_vals,model_snr,plotopts,'DisplayName',display_name,'MarkerSize',12,'MarkerFaceColor',plotopts(1));
%lines = findobj(gca,'Type','line');
%lines(1).MarkerFaceColor=lines(end).Color;
    
title('Overall model SNR','interpreter','latex');
xlabel('$h/r_s$ (ms)','interpreter','latex');
ylabel('SNR (dB)','interpreter','latex');
set(gca,'FontSize',fontsize);
set(gca,'TickLabelInterpreter','latex');
legend('interpreter','latex');


%ylim([0 18]);
%xlim([0.0025 0.045]);

% FIGURE 2: retraining time
    
if online_training
    if ~exist('retraining_fig','var')
        retraining_fig=figure;
    else
        figure(retraining_fig);
    end
    hold on;

    if ~fixed_point
        display_name = hidden_size+" hidden neurons";
        if hidden_size==10
           plotopts = 'rs-';
        elseif hidden_size==20
           plotopts = 'gd-';
        elseif hidden_size==30
            plotopts = 'b^-';
        elseif hidden_size==40
            plotopts = 'mv-';
        else
            plotopts = 'cx-';
        end
    else
        display_name = wordsize+" bits";
        if wordsize==4
           plotopts = 'rs-';
        elseif wordsize==5
            plotopts = 'gd-';
        elseif wordsize==6
            plotopts = 'b^-';
        elseif wordsize==7
            plotopts = 'mv-';
        else
            plotopts = 'cx-';
        end
    end
    
    plot(x_vals,conv_times .* 1e3,plotopts,'DisplayName',display_name,'MarkerSize',12,'MarkerFaceColor',plotopts(1));
    %lines = findobj(gca,'Type','line');
    %lines(end).MarkerFaceColor=lines(end).Color;
    
    title('Retraining time','interpreter','latex');
    xlabel('$h/r_s$ (ms)','interpreter','latex');
    ylabel('retraining time (ms)','interpreter','latex');
    set(gca,'FontSize',fontsize);
    set(gca,'TickLabelInterpreter','latex');
    legend('interpreter','latex');
end

hold off;

%end


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

