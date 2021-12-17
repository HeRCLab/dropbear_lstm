function signal_pred_puja = perform_fft_forecast (x,x_sub,signal,signal_sub,model_sample_rate,fft_window,fft_step,prediction_time,nonstationarity_time)
    
    detrend = 0;

    % plot spectum of signal
    spectrum = fft(signal_sub);
    spectrum = spectrum(1:floor(numel(spectrum)/2));
    spectrum = abs(spectrum);
    freqs = (1:numel(spectrum)) .* ((model_sample_rate/2)/numel(spectrum));
    plot(freqs,spectrum);
    hold on;
    xlabel('Hz');
    ylabel('Power');
    title('Spectrum of subsampled input signal');
    hold off;

    % apply Puja approach

    % compute size of output
    num_samples = numel(signal_sub);
    % allocate output
    signal_pred_puja = zeros(size(signal_sub));

    for i = fft_window:fft_step:num_samples-fft_window+1
        block = signal_sub(i-fft_window+1:i);
        
        % detrend the signal
        %g = fittype('a*x+b');
        %model = fit((1:numel(block))',block',g);
        %trend = (1:numel(block)) * model.a + model.b;
        
        if detrend
            X = [ones(length(block),1) (1:numel(block))'];
            b = X\(block');
            trend = (1:numel(block)) * b(2) + b(1);
            block = block - trend;
        end
        
        % compute fft
        the_fft = fft(block);
        
        % use ifft to predict next window

        % adjust the phase of each frequency
        for n=2:fft_window/2
            re = real(the_fft(n));
            im = imag(the_fft(n));
            ma = abs(the_fft(n));
            an = atan2(im,re);

            period = 1/((n-1)/fft_window * model_sample_rate);
            needed_phase_shift_time = fft_window / model_sample_rate + prediction_time / model_sample_rate;
            needed_phase_shift_radians = needed_phase_shift_time/period * 2*pi;
            
            % testing code
            %needed_phase_shift_radians = needed_phase_shift_radians;
            %needed_phase_shift_radians = 0;
            
            val = ma * exp(1i*(an + needed_phase_shift_radians));
            the_fft(n) = val;
            if n>1
                the_fft(fft_window-n+2) = conj(val);
            end
        end

        % perform IFFT and re-trend
        the_ifft = ifft(the_fft);
        
        if detrend
            the_ifft = the_ifft + trend;
        end

        % shift by prediction time
        %the_ifft = the_ifft(phase_shift+1:end);

        % incorporate into forecast
        signal_pred_puja(i:i+fft_window-1) = the_ifft;
    end

    % determine the actual forecast time
    min_error = 1e10;
    error_trend = zeros(1,fft_window+1);
    for shamt = -fft_window/2:fft_window/2
        if shamt < 0
            shifted = [signal_pred_puja(-shamt+1:end) zeros(1,-shamt)];
        elseif shamt > 0
            shifted = [zeros(1,shamt),signal_pred_puja(1:end-shamt)];
        else
            shifted = signal_pred_puja;
        end
        error = mean((signal_sub - shifted).^2)^.5;
        if error < min_error
            min_error = error;
            best_offset = shamt;
        end
        error_trend(shamt+fft_window/2+1) = error;
    end

    best_offset

    figure;
    plot(-fft_window/2:fft_window/2,error_trend);
    hold on;
    xlabel('shamt');
    ylabel('rms error');
    hold off;

    phase_shift = best_offset;
    %phase_shift = 0;

    % phase shift to line up with signal
    if phase_shift < 0
        signal_pred_puja = [signal_pred_puja(-phase_shift+1:end),zeros(1,-phase_shift)];
    else
        signal_pred_puja = [zeros(1,phase_shift),signal_pred_puja(1:end-phase_shift)];
    end
    
    % plot Puja forecaster against subsampled input
    hold off;
    figure;
    plot(x_sub(1:numel(signal_sub)),signal_sub);
    hold on;
    plot(x_sub(1:numel(signal_pred_puja)),signal_pred_puja);
    xlabel('time (s)');
    legend({'original subsampled','puja prediction'});

    % plot Puja signal

    % plot Puja forecaster against original input
    hold off;
    figure;
    plot(x,signal);
    hold on;
    signal_pred_puja_zoh = myzoh(x,x_sub,signal_pred_puja);
    plot(x,signal_pred_puja_zoh);
    xlabel('time (s)');
    legend({'original','puja prediction'});
    hold off;

    % error of prediction at native sample rate and subsample rate
    % only calculate SNR for after nonstationarity (second half)
    
    error_sub = signal_sub(1:numel(signal_pred_puja)) - signal_pred_puja;
    error = signal_pred_puja_zoh - signal;

    % isolate the portion of the signal after the nonstationarity
    half_signal_point = find(x>=nonstationarity_time);
    half_signal_point = half_signal_point(1);
    
    half_signal_point_sub = find(x_sub>=nonstationarity_time);
    half_signal_point_sub = half_signal_point_sub(1);
    
    error_power = rms(error(half_signal_point+1:end))^2;
    error_sub_power = rms(error_sub(half_signal_point_sub+1:end))^2;
    
    % calculate SNR of Puja approach vs original signal
    signal_power = rms(signal_pred_puja_zoh(half_signal_point+1:end))^2;
    signal_sub_power = rms(signal_pred_puja(half_signal_point_sub+1:end))^2;
        
    puja_snr = log10(signal_power / error_power) * 20
    puja_sub_snr = log10(signal_sub_power / error_sub_power) * 20
    
    % plot errors
    figure;
    plot(x_sub,error_sub(1:numel(x_sub)));
    hold on;
    title('error');
    xlabel('time (s)');
    legend({'error vs subsampled'});
    drawnow;

    % plot errors
    figure;
    plot(x,error);
    hold on;
    title('error');
    xlabel('time (s)');
    legend({'error vs original'});
    drawnow;

    

    % calculate convergence time of Puja approach vs original signal
    [~,conv_time] = get_accuracy_stats (x,signal,signal_pred_puja_zoh,error,nonstationarity_time,1)

end
