% this script is my sandbox for testing low-level and online training

% close all the plots from previous run
close all;
clear all;

% debug
global output_biases;
output_biases=[];

% clear errors
errors=[];

% initialize empty frames
frames=[];

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
time = 0.3;
amps=[10,4,7,2,8,5];
freqs=[10,37,78,5,67,75];
phases=[0,0,0,0,0,0];

% synthesize signal
[x,signal] = make_signal(sample_rate,time,amps,freqs,phases);

% read Puga's signal
%data=lvm_import('Ivol_Acc_Load_1S_1STD.lvm');
%x = data.Segment1.data(:,1);
%signal = data.Segment1.data(:,4)';
%sample_rate = numel(data.Segment1.data(:,1))/data.Segment1.data(end,1);

% training parameters
model_sample_rate = 1000;
prediction_time = 10;
subsample = floor(sample_rate / model_sample_rate);

% select network type
history_length = 10; % for MLP

% synthesize subsampled signal
% NOTE: this uses striding, not interpolation as in the C-based model!
x_sub = x(1:subsample:end);
signal_sub = signal(1:subsample:end);

% plot subsampled signal
myfig2 = figure('Position', [20 50 1650 800]);
hold on;
subplot(4,1,1);
plot(x_sub,signal_sub);
title('subsampled signal');
xlabel('time (s)');

subplot(4,1,2);

% assume i is the current sample being read from the accelerometer, so
% we are predicting sample:
%   i+prediction_time
% based on:
% samples i-history_length-prediction_time to i-prediction_time

signal_pred = [];

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
        [mynet,output_from_mlp] = build_ann(x_train,y_train,[10],epochs);
        first_training=0;
    else
        [mynet,output_from_mlp] = build_ann(x_train,y_train,[10],epochs,mynet);
    end
    
    % predict next window
    predicted = mypredict(mynet,signal_sub(i-history_length+1:i));
    signal_pred=[signal_pred,predicted];

    error = signal_pred(end) -...
                signal_sub(i+prediction_time);

    rmse = mean(error.^2)^.5;
    
    fprintf('rms of segment training window %d = %0.4e\n',floor((i-1)/1)+1,rmse);
    subplot(4,1,2);
    plot((1:size(signal_pred,2))./model_sample_rate,signal_pred-signal_sub(1:size(signal_pred,2)),'r');
    title('RMS error');
    xlabel('time (s)');
    ylabel('rmse');
    
    % plot predicted signal
    subplot(4,1,3);
    time_axis = 0:(numel(signal_pred)-1);
    time_axis = time_axis ./ model_sample_rate;
    plot(time_axis,signal_sub(1,1:size(time_axis,2)),'r');
    hold on;
    plot(time_axis,signal_pred,'b');
    xlabel('time (s)');
    title('predicted signal');
    
    % plot output bias
    subplot(4,1,4);
    plot(output_biases);
    xlabel('sample');
    title('output bias');
    
    %pause(1e-6);
    drawnow;
    frames=[frames;getframe(gcf)];
    
    %figure;
    %plot(x_sub,signal_pred);
    
end

writerObj = VideoWriter('myVideo.mp4','MPEG-4');%VideoCompressionMethod,'H.264');
writerObj.Quality=100;
writerObj.FrameRate = 10;
%writerObj.VideoBitsPerPixel=24;
open(writerObj);
for i=1:length(frames)
    % convert the image to a frame
    frame = frames(i) ;    
    writeVideo(writerObj, frame);
end
close(writerObj);

error = signal_sub(1,1:size(signal_pred,2))-...
         signal_pred;

subplot(4,1,4);
rmse = mean(error.^2)^.5;
rmse = error;
str = sprintf('predicted signal, rms = %0.4e',rmse);
title(str);
%ylim([min_val,min_val+range]);
