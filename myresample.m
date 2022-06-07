function [resampled_x,resampled_y] = myresample(signal,sample_rate,subsample_rate)

    % compute sample ratio (e.g. 0.5 is half-rate)
    subsample_ratio = subsample_rate/sample_rate;
    
    % compute length of new signal
    len_new = ceil(numel(signal) * subsample_ratio);
    
    % allocate new signal
    resampled_x = single(zeros(1,len_new));
    resampled_y = single(zeros(1,len_new));
    
    signal_len = numel(signal);
    
    for i=1:len_new
        position = (i-1) / subsample_ratio;
        position_frac = position - floor(position);
        position_int = floor(position);
        
        current_sample = position_int+1;
        next_sample = position_int+2;
        if next_sample > signal_len
            next_sample = current_sample;
        end 
        
        resampled_y(1,i) = (1-position_frac)*signal(1,current_sample) +...
                            (position_frac)*signal(1,next_sample);
        resampled_x(1,i) = (i-1) / subsample_rate;
    end
    
end
