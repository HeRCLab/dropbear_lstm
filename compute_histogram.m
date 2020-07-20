function [hist,y0,y1,bin_size] = compute_histogram(y)

    % find bin width, based on largest difference between samples
    
    % compute adjacency matrix (for first column if two)
    am = zeros(1,numel(y(:,1)));
    for i = 1:numel(y(:,1))
        min_distance = 1e6;
        for j = 1:numel(y(:,1))
            d = abs(y(i)-y(j));
            if d<min_distance && i~=j
                min_distance=d;
            end
        end
        am(i) = min_distance;
    end
    
    % find max distance between any two points
    bin_size = max(am);
    
    % allocate bins
    y0 = min(min(y));
    y1 = max(max(y));
    bins = ceil((y1 - y0) / bin_size);
    if size(y,2)==1
        % 1D histogram for 1D data
        hist = zeros(1,bins);
    else
        % 2D histogram for 1D data
        hist = zeros(bins,bins);
    end
    
    % count occurances
    for i=1:numel(y(:,1))
        bin = floor((y(i,1) - y0) / bin_size)+1;
        if size(y,2)==1
            % 1D
            hist(1,bin) = hist(1,bin) + 1;
        else
            % 2D
            if y(i,1) == -0.0121
                1;
            end
            if i == 20
                1;
            end
            
            
            bin2 = floor((y(i,2) - y0) / bin_size) + 1;
            hist(bin,bin2) = hist(bin,bin2) + 1;
        end
    end
    
    % normalize
    hist = hist ./ numel(y(:,:));
    
%     x=[y0:(y1-y0)/bins:y1];
%     x=x(1,1:bins);
%     if size(y,2) == 1
%         figure;
%         bar(x,hist);
%     else
%         figure;
%         surf(x,x,hist);
%     end
    
end
