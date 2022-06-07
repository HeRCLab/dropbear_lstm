function [snr,conv_time] = get_accuracy_stats (x,signal,signal_pred_zoh,error,nt_time,plotit)
    
    % SNR
    signal_power = rms(signal)^2;
    error_power = rms(error)^2;
    snr = log10(signal_power / error_power) * 20;

    % convergence
    g = fittype('a-b*exp(-c*x)');

    % split the time scale
    half_signal_point = find(x>=nt_time);
    half_signal_point = half_signal_point(1);
    x1 = x(1:half_signal_point);
    x2 = x(half_signal_point+1:end);
    
    error_signal_rms = abs(error);

    % split the error signal
    half1 = error_signal_rms(1:half_signal_point);
    half2 = error_signal_rms(half_signal_point+1:end);

    % assume x2 has more points, so trim error
    x2 = x2(1:numel(x1));
    half2 = half2(1:numel(x1));

    % fit errors starting at t=0
    model1 = fit(x1',half1',g);
    % start fitting the second error curve at t=0 (use x1)
    model2 = fit(x1',half2',g,'Lower',[0,-Inf,0],'Upper',[Inf,0,Inf]);

    % build error curves starting at t=0
    error_curve1 = model1.a-model1.b*exp(-model1.c.*x1);
    error_curve2 = model2.a-model2.b*exp(-model2.c.*x1);

    conv_time=-model2.b/model2.c^2;
    
    if plotit
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
        title("FFT");

        subplot (2,1,2);
        %plot(x,error_signal,'r');
        plot(x,error_signal_rms,'r');
        hold on;
        legend({'error signal rms'});

        xlabel('time');
        ylabel('accel');
        title("predicted signal");
        
        %plot(x1,error_curve1,'g');
        plot(x2,error_curve2,'g');
    	%legend({'error signal rms','error curve1','error curve2'});
        legend({'error signal rms','error fit'});
        
        drawnow;
        hold off;
        
    end
    
end
