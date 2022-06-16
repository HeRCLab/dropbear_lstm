function [snr,conv_time,a,b,c] = get_accuracy_stats (x,signal,signal_pred_zoh,error,nt_time,plotit,h,title_prefix)
    
    % parameters
    fontsize = 16;
        
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

%     % assume x2 has more points, so trim error
%     if numel(x1) > numel(x2)
%         x1 = x1(1:numel(x2));
%         half2 = half2(1:numel(x2));
%     elseif numel(x1) > numel(x2)
%         x2 = x2(1:numel(x1));
%         half2 = half2(1:numel(x1));
%     end
    
    % fit errors starting at t=0
    model1 = fit(x1',half1',g);
    % start fitting the second error curve starting at t=0
    t2 = x2'-x2(1);
    %t2 = t2(1:h);
    %y2 = half2(1:h);
    y2 = half2;
    
    model2 = fit(t2,y2',g,'Lower',[0,-Inf,0],'Upper',[Inf,0,Inf]);

    % build error curves starting at t=0
    error_curve1 = model1.a-model1.b*exp(-model1.c.*x1);
    error_curve2 = model2.a-model2.b*exp(-model2.c.*(x2'));

    %conv_time=-model2.b/model2.c^2;
    conv_time=-log(0.5)/model2.c;
    
    a = model2.a;
    b = -model2.b;
    c = model2.c;    
    
    if plotit
        % plot signal and error
        figure;
        subplot (2,1,1);
        plot(x,signal,'b');
        xlim([9 9.05]);
        hold on;
        plot(x,signal_pred_zoh,'r');
        %plot(x,error_smooth,'g');
        legend({'$V(t)$','$V_{forecast}(t-f/r_s)$'},'interpreter','latex');
        xlabel('$t$','interpreter','latex');
        %ylabel('acceleration','interpreter','latex');
        title(title_prefix+": Results",'interpreter','latex');
        set(gca,'FontSize',fontsize);
        set(gca,'TickLabelInterpreter','latex')
        
        subplot (2,1,2);
        %plot(x,error_signal,'r');
        plot(x,error_signal_rms,'r');
        xlim([9.6 10.5]);
        hold on;
        %legend({'abs(noise signal)'});

        xlabel('$t$','interpreter','latex');
        %ylabel('accel','interpreter','latex');
        title(title_prefix+": Noise",'interpreter','latex');
        
        %plot(x1,error_curve1,'g');
        plot(x2',error_curve2,'g','LineWidth',3);
    	%legend({'error signal rms','error curve1','error curve2'});
        legend({'$|N(t)|$','$fit(|N(t)|)$'},'interpreter','latex');
        
        set(gca,'FontSize',fontsize);
        set(gca,'TickLabelInterpreter','latex');
        drawnow;
        hold off;
        
    end
    
end
