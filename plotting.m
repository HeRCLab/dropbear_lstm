close all;

history = [25:25:250]./2500.*1000;
plot(history,1./conv_rates_2500Hz_100hidden,'o-');
hold on;
xlabel('Model history length (ms)');
ylabel('Convergence time (s)');
title('Model convergence time');
plot(history,1./conv_rates_2500Hz_8bit_100hidden,'o--');
plot(history,1./conv_rates_5000Hz_100hidden,'s-');
plot(history,1./conv_rates_5000Hz_8bit_100hidden,'s--');
legend({"2500 Hz model (float32)","2500 Hz model (8-bit fixed)","5000 Hz model (float32)","5000 Hz model (8-bit fixed)"});

figure
plot(history,conv_snrs_2500Hz_100hidden,'go-');
hold on;
plot(history,conv_snrs_2500Hz_8bit_100hidden,'gs-');
plot(history,subsample_snr_2500Hz_100hidden,'g:');
plot(history,subsample_snr_2500Hz_8bit_100hidden,'g--');

plot(history,conv_snrs_5000Hz_100hidden,'bo-');
plot(history,conv_snrs_5000Hz_8bit_100hidden,'bs-');
plot(history,subsample_snr_5000Hz_100hidden,'b:');
plot(history,subsample_snr_5000Hz_8bit_100hidden,'b--');
xlabel('Model history length (ms)');
ylabel('SNR (dB)');
title('Model accuracy');
legend({"2500 Hz model (float32)","2500 Hz model (8-bit fixed)","2500 Hz subsample ceiling (float32)","2500 Hz subsample ceiling (8-bit fixed)","5000 Hz model (float32)","5000 Hz model (8-bit fixed)","5000 Hz subsample ceiling (float32)","5000 Hz subsample ceiling (8-bit fixed)"});

%conv_rates_2500Hz_100hidden = conv_rates
%subsample_snr_2500Hz_100hidden = subsample_snr
%conv_snrs_2500Hz_100hidden = conv_snrs2

% conv_rates_2500Hz_8bit_100hidden = conv_rates
% subsample_snr_2500Hz_8bit_100hidden = subsample_snr
% conv_snrs_2500Hz_8bit_100hidden = conv_snrs2
