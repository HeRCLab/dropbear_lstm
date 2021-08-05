close all;

history = [25:25:250]./2500.*1000;
%plot(history,1./conv_rates_2500Hz_6bit_10hidden,'o-');
plot(history,conv_rates_2500Hz_6bit_20hidden*1000,'o-');
hold on;
xlabel('Model history length (ms)');
ylabel('Convergence time (ms)');
title('Model convergence time');
plot(history,conv_rates_2500Hz_6bit_30hidden*1000,'+-');
plot(history,conv_rates_2500Hz_6bit_40hidden*1000,'s-');
plot(history,conv_rates_2500Hz_6bit_50hidden*1000,'*-');

%plot(history,1./conv_rates_2500Hz_100hidden,'s-');
%plot(history,1./conv_rates_2500Hz_6bit_100hidden,'s--');
%legend({"2500 Hz model (float32)","2500 Hz model (8-bit fixed)","2500 Hz model (float32)","2500 Hz model (8-bit fixed)"});
%legend({"2500 Hz model (8-bit fixed)","2500 Hz model (7-bit fixed)","2500 Hz model (6-bit fixed)"});
legend({"20 hidden","30 hidden","40 hidden","50 hidden"});

figure
plot(history,conv_snrs_2500Hz_6bit_20hidden,'o-');
hold on;
plot(history,conv_snrs_2500Hz_6bit_30hidden,'+-');
plot(history,conv_snrs_2500Hz_6bit_40hidden,'s-');
plot(history,conv_snrs_2500Hz_6bit_50hidden,'*-');
plot(history,subsample_snr_2500Hz_6bit_30hidden,'b--');

xlabel('Model history length (ms)');
ylabel('SNR (dB)');
title('Model accuracy');
legend({"20 hidden","30 hidden","40 hidden","50 hidden","subsampling/quantization SNR"});

% effect of precision
figure
plot(history,conv_snrs_2500Hz_5bit_20hidden,'o-');
hold on;
plot(history,conv_snrs_2500Hz_6bit_20hidden,'+-');
plot(history,conv_snrs_2500Hz_7bit_20hidden,'s-');
plot(history,conv_snrs_2500Hz_8bit_20hidden,'*-');

% plot(history,subsample_snr_2500Hz_5bit_20hidden,'o--');
% plot(history,subsample_snr_2500Hz_6bit_20hidden,'+--');
% plot(history,subsample_snr_2500Hz_7bit_20hidden,'s--');
% plot(history,subsample_snr_2500Hz_8bit_20hidden,'*--');

xlabel('Model history length (ms)');
ylabel('SNR (dB)');
title('Model accuracy, 20 hidden neurons, 2500 Hz');
legend({"5-bit fixed","6-bit fixed","7-bit fixed","8-bit fixed"});
        
figure
plot(history,conv_rates_2500Hz_5bit_20hidden*1000,'o-');
hold on;
plot(history,conv_rates_2500Hz_6bit_20hidden*1000,'+-');
plot(history,conv_rates_2500Hz_7bit_20hidden*1000,'s-');
plot(history,conv_rates_2500Hz_8bit_20hidden*1000,'*-');

% plot(history,subsample_snr_2500Hz_5bit_20hidden,'o--');
% plot(history,subsample_snr_2500Hz_6bit_20hidden,'+--');
% plot(history,subsample_snr_2500Hz_7bit_20hidden,'s--');
% plot(history,subsample_snr_2500Hz_8bit_20hidden,'*--');

xlabel('Model history length (ms))');
ylabel('Convergence time (ms)');
title('Convergence time, 20 hidden neurons, 2500 Hz');
legend({"5-bit fixed","6-bit fixed","7-bit fixed","8-bit fixed"});

conv_rates_2500Hz_5bit_20hidden = conv_rates
subsample_snr_2500Hz_5bit_20hidden = subsample_snr
conv_snrs_2500Hz_5bit_20hidden = conv_snrs2

conv_rates_2500Hz_6bit_30hidden = conv_rates
subsample_snr_2500Hz_6bit_30hidden = subsample_snr
conv_snrs_2500Hz_6bit_30hidden = conv_snrs2

conv_rates_2500Hz_6bit_40hidden = conv_rates
subsample_snr_2500Hz_6bit_40hidden = subsample_snr
conv_snrs_2500Hz_6bit_40hidden = conv_snrs2

conv_rates_2500Hz_6bit_50hidden = conv_rates
subsample_snr_2500Hz_6bit_50hidden = subsample_snr
conv_snrs_2500Hz_6bit_50hidden = conv_snrs2
