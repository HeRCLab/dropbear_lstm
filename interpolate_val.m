function interpolate_val = interpolate_val(x,y,x1)

    % find position of x1 within array x
    for i = 1:numel(x)-1
        if x(i) > x1
            idx = max([i-1,1]);
            break;
        end
    end

    % find lower and upper samples
    floor_val = y(idx);
    ceil_val = y(idx+1);
    
    % compute original sample period
    delta_t = x(idx+1) - x(idx);
    
    % compute weight
    weight_floor = (x1 - x(idx))/delta_t;
    weight_ceil = (x(idx+1) - x1)/delta_t;
    
    % bilinear interpolate
    interpolate_val = weight_floor * floor_val + weight_ceil * ceil_val;
    
end
