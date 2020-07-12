% this script is my sandbox for testing low-level and online training
% close all the plots from previous run
close all;
clear all;
profile on

% seed RNG
rng(42);

% parameters

% signal synthesis
sample_rate = 5000;
time = 60;
amps=[1, 2, 3, 0.7, 2.3, 1];
freqs=[2, 3.7, 7.80, 0.12, 0.54, 1.3];
phases=[0, 1, 2, 3, 5, 1];

% synthesize signal
[x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

% read Puga's signal
%data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
%x = data.Segment1.data(:,1);
%signal = data.Segment1.data(:,4)';
%sample_rate = numel(data.Segment1.data(:,1))/data.Segment1.data(end,1);

% model and training parameters
history_length = 10;
hidden_size = 10;
model_sample_rate = 1250;
subsample = floor(sample_rate / model_sample_rate);
prediction_time = 10;
alpha = 1;

% synthesize subsampled signal
% NOTE: this uses striding, not interpolation as in the C-based model!
sample_period = 1/model_sample_rate;
[x_sub,signal_sub] = myresample(signal,sample_rate,model_sample_rate);

% reuse network across trainings; otherwise, start from randomized
% network each training
reuse_network = 1;

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

% epochs
epochs=1;

myfig2 = figure('Position', [20 50 1650 800]);
hold on;

% assume i is the current sample being read from the accelerometer, so
% we are predicting sample:
%   i+prediction_time
% based on:
% samples i-history_length-prediction_time to i-prediction_time

signal_pred = [];
rms_error_aligned = [];
rms_error_nonaligned = [];

signal_pred = zeros(size(signal_sub));

for i = 1:numel(signal_sub)-prediction_time
    start_idx = i-history_length-prediction_time;
    
    if start_idx < 1
        signal_pred=[signal_pred,0];
        continue;
    end

    x_train = signal_sub(start_idx:start_idx+history_length-1);
    y_train = signal_sub(i);
    
    % train network
    if first_training
        [mynet,output_from_mlp] = build_ann(x_train,y_train,[hidden_size],epochs,alpha);
        first_training=0;
    else
        [mynet,output_from_mlp] = build_ann(x_train,y_train,[10],epochs,alpha,mynet);
    end
    
    x_pred = signal_sub(i-history_length+1:i);
    signal_pred(1,i+prediction_time)=mypredict(mynet,x_pred);

    %pause(1e-6);
    if mod(i,50)==0
        % plot predicted signal
        p=signal_pred(prediction_time:i-1);
        s=signal_sub(1:i-prediction_time);
        time_axis = 0:(size(p,2)-1);
        time_axis = time_axis ./ model_sample_rate;

        subplot(4,1,1);        
        plot(time_axis,s,'r');
        hold on;
        plot(time_axis,p,'b');
        xlabel('time (s)');
        title('subsampled signal, predicted signal');
        legend ('signal (actual)','signal (predicted)','Location','northwest');
        
        subplot(4,1,2);
        inst_error = s-p;
        plot(time_axis,inst_error,'r');
        title('instantaneous error with shifted prediction');
        xlabel('time (s)');
        ylabel('difference');

        if i>(2*prediction_time+history_length)
            subplot(4,1,3);
            
            p=p(history_length+1:end);
            s=s(history_length+1:end);
            
            rmse = mean((p-s).^2)^.5;
            rms_error_aligned=[rms_error_aligned,rmse];
            
            time_axis = 0:(size(rms_error_aligned,2)-1);
            time_axis = time_axis ./ model_sample_rate;
            
            plot(time_axis,rms_error_aligned,'r');
            title("cumulative rms error after shifting predicted signal "+prediction_time+" samples");
            xlabel('time (s)');
            ylabel('rmse');
        end

        % plot output bias
        subplot(4,1,4);
        plot(deltas_output);
        xlabel('sample');
        title('output delta');

        % plot signal overlapped with predicted signal
%        subplot(4,1,5);
%         s = signal_sub((prediction_time+1):size(signal_pred,2));
%         %s = signal_sub(1:size(signal_pred,2));
%         p = signal_pred(1:end-prediction_time);
%         %p = signal_pred;
%         time_axis = 0:(numel(s)-1);
%         time_axis = time_axis ./ model_sample_rate;
%         plot(time_axis,s,'r');
%         hold on;
%         plot(time_axis,p,'b');
%         xlabel('time (s)');
%         title('subsampled signal, shifted predicted signal');
%         legend ('signal (actual)','signal (predicted)','Location','northwest');
        drawnow;
    end
    %frames=[frames;getframe(gcf)];
    
    %figure;
    %plot(x_sub,signal_pred);
    
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

