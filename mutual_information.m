function mutual_information = mutual_information (x,y,tau)
    % compute Eq. 9 from Simon's paper
    
    % assume that (x,y) are the original (maximum) sample rate

    % compute prob distribution on original signal
    [hist,y0,y1,bin_size] = compute_histogram (y');
    
    % compute joint probability
    % determine shift amount
    original_sample_period = x(2)-x(1);
    shift_amount = floor(tau/original_sample_period);
    y_joint = [y(1:end-shift_amount)',y(shift_amount+1:end)'];
    [hist_joint,y0,y1,bin_size] = compute_histogram (y_joint);
    
    sum = 0;
    for i=shift_amount+1:shift_amount:numel(y)
        % compute probability of k
        bin1 = floor((y(i)-y0)/bin_size)+1;
        prob_k = hist(bin1);
        
        % compute probability of k-tau
        bin2 = max([1,floor((y(i-shift_amount)-y0)/bin_size)+1]);
        prob_k_minus_tau = hist(bin2);

        joint = hist_joint(bin2,bin1);
        if joint == 0
            1;
        end
        
        val = joint * log(joint/(prob_k*prob_k_minus_tau));
        if isnan(val)
            1;
        end
        sum = sum + val;
    end

    mutual_information = sum;
    
end
