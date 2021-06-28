function [resampled_x,resampled_y] = myresample(signal,sample_rate,subsample_rate)

    subsample_ratio = subsample_rate/sample_rate;
    len_new = min([ceil(numel(signal) * subsample_ratio) numel(signal)-1]);
    resampled_x = single(zeros(1,len_new));
    resampled_y = single(zeros(1,len_new));
    
    for i=1:len_new
        position = (i-1) / subsample_ratio;
        position_frac = position - floor(position);
        position_int = floor(position);
        resampled_y(1,i) = (1-position_frac)*signal(1,position_int+1) +...
                            (position_frac)*signal(1,position_int+2);
        resampled_x(1,i) = (i-1) / subsample_rate;
    end
    
end
