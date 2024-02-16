function [] = main ()
    
%     n_points=6;
%     n_homologies = n_points + (n_points^2-n_points)/2 + ...
%                 (n_points^3-3*n_points^2+2*n_points)/6;
%     for i = 1:n_homologies
%         find_points_from_enumeration (i,n_points)
%     end
%     return;

    % set up parameters
    n = 1; % number of longest persistances
    sample_rate = 1000;
    dim = 2;
    use_higher_sample_rate_for_inputs = 0;
    radius = 1e6;
    tda_window = .01; % in seconds
    
    % read and plot data
    [time_vibration,vibration_signal,...
            time_pin,pin_position,vibration_signal_samples,start_time] = read_data (sample_rate,dim,1,5.5);
    
    % extract a window for TDA
    number_of_samples_in_window = sample_rate * tda_window;
    
    % only compute TDA features for the first window
    % note this will eventually be a loop to compute all windows
    n_windows = (size(vibration_signal_samples,1)-number_of_samples_in_window+1);

    % overridden to perform a partial run
    n_windows = 5.5 * sample_rate;

    longest_distances = zeros(n_windows,n);
    for window_number = 1:n_windows
        %sample_range_in_window = (window_number-1)*number_of_samples_in_window+1:window_number*number_of_samples_in_window;
        sample_range_in_window = window_number:window_number+number_of_samples_in_window-1;
        window_data = vibration_signal_samples(sample_range_in_window,:);

        % perform TDA
        [distance_matrix,boundary_matrix,all_h,n_h1,n_h2] = tda (window_data,radius);
        % extract features
        longest_distances(window_number,:) = longest_persistance(distance_matrix,boundary_matrix,n,n_h1,all_h);
        if mod(window_number,100)
            plot((1:window_number)./sample_rate+start_time,longest_distances(1:window_number),'r');
            hold on;
            drawnow;
        end
    end

    xlabel('time (s)');
    ylabel('longest persistance');

    figure new;
    plot(time_pin(1:window_number),pin_position(1:window_number));
    xlabel('time(s)');
    ylabel('displacement (m)');

end

function longest_distances = longest_persistance(distance_matrix,boundary_matrix,n,n_h1,all_h)

    longest_distances = zeros(1,n);
    longest_enums = zeros(1,n);

    n_points = size(distance_matrix,1);
    n_h = max(cell2mat(all_h(:,3)));

    % iterate over all H0, H1, and H2 homologies
    for i = 1:n_h
        death = find_death(boundary_matrix,distance_matrix,i,n_points,n_h1,all_h);
        birth = find_birth(boundary_matrix,distance_matrix,i,n_points,all_h);
        persistance = death - birth;

        for j=1:n
            if persistance > longest_distances(j)
                longest_distances = [longest_distances(1,1:j-1),persistance,longest_distances(1,j:n-1)];
                longest_enums = [longest_enums(1,1:j-1),i,longest_enums(1,j:n-1)];
                break;
            end
        end
    end

end

% find the birth time of a given homology
function birth_time = find_birth (boundary_matrix,distance_matrix,enum,n_points,all_h)
    % find the points for this homology
    points = find_points_from_enumeration (enum,n_points,all_h);

    if numel(points)==1
        birth_time=0;
    elseif numel(points)==2
        birth_time=distance_matrix(points(1),points(2));
    else
        birth_time = max([distance_matrix(points(1),points(2)),...
            distance_matrix(points(1),points(3)),...
            distance_matrix(points(2),points(3))]);
    end
end

% find the death time of a given homology
% assume a time horizon of the largest distance
function death_time = find_death (boundary_matrix,distance_matrix,enum,n_points,n_h1,all_h)

    if enum > n_points + n_h1
        % H2's always die at the horizon
        death_time = max(max(distance_matrix));
        return;
    else
        % check to see if this enum was terminated by another
        % for each column with a 1 for row 'enum'...
        overriding_homology = -1;
        for col_num = find(boundary_matrix(enum,:))
            % check if the row corresponds to the bottom 1
            if max(boundary_matrix(enum+1:end,col_num)) == 0
                overriding_homology = col_num;
                break;
            end
        end
    end

    if overriding_homology == -1
        death_time = max(max(distance_matrix));
        return;
    end

    % find the points for this homology
    points = find_points_from_enumeration (overriding_homology,n_points,all_h);

    assert(numel(points)>1);

    if numel(points) == 2
        % H1
        death_time = distance_matrix(points(1),points(2));
    else
        % H2
        death_time = max([distance_matrix(points(1),points(2)),...
            distance_matrix(points(1),points(3)),...
            distance_matrix(points(2),points(3))]);
    end
end

% converts a scalar index into a set of points
function points = find_points_from_enumeration (enum,n_points,all_h)
   for i=1:size(all_h,1)
       if all_h{i,3} == enum
           points = all_h{i,2};
           break;
       end
   end
end

% converts a set of points into a scalar index
function enum = find_enumeration_from_points (points,n_points,all_h)
    for i=1:size(all_h,1)
       if all_h{i,2} == points
           num = all_h{i,3};
           break;
       end
    end
end

function [h0,h1,h2] = distance_matrix_to_list (dist_matrix)
    n = size(dist_matrix,1);
    n_pairs = (n^2-n)/2;
    n_triples = (n^3 - 3*n^2 + 2*n)/6;

    h = cell(n,3);
    h1 = cell(n_pairs,3);
    h2 = cell(n_triples,3);

    for i=1:n
        h0{i,1} = 0;
        h0{i,2} = i;
        h0{i,3} = i;
    end

    cnt=1;
    for i=1:n-1
        for j=i+1:n
            h1{cnt,1} = dist_matrix(i,j);
            h1{cnt,2} = [i,j];
            h1{cnt,3} = 0;
            cnt=cnt+1;
        end
    end

    cnt=1;
    for i=1:n-2
        for j=i+1:n-1
            for k=j+1:n
                h2{cnt,1} = max([dist_matrix(i,j),dist_matrix(i,k),dist_matrix(j,k)]);;
                h2{cnt,2} = [i,j,k];
                h2{cnt,3} = 0;
                cnt=cnt+1;
            end
        end
    end

end

function [h,n_h] = find_homologies(h,distance_matrix)
    n_h = 0;
    used = zeros(1,size(distance_matrix,1));
    for i=1:size(h,1)
        nodes = h{i,2};
        used_node = 0;
        for node = nodes
            if used(node)
                used_node = 1;
                break;
            end
        end
        if ~used_node
            n_h = n_h + 1;
            h{i,3} = n_h;
            used(nodes) = 1;
        end
    end
end

% return boundary matrix and distance matrix for input data
function [distance_matrix,boundary_matrix,all_h,n_h1,n_h2] = tda (window_data,radius)
    % fill in distance matrix
    distance_matrix = zeros(size(window_data,1),size(window_data,1));
    for i=1:size(window_data,1)
        for j=1:size(window_data,1)
            distance_matrix(i,j) = norm(window_data(i,:)-window_data(j,:));
        end
    end
    
    boundary_list = {};
    
    % real method???
    % find the earliest birth of H-1's
    % mark all vertices as unused
    [h0,h1,h2] = distance_matrix_to_list (distance_matrix);
    h1 = sortrows(h1,1);
    h2 = sortrows(h2,1);

    n_h0 = size(distance_matrix,1);

    [h1,n_h1] = find_homologies(h1,distance_matrix);
    for i=1:size(h1,1)
        if h1{i,3}
            h1{i,3} = h1{i,3} + n_h0;
        end
    end

    [h2,n_h2] = find_homologies(h2,distance_matrix);
    for i=1:size(h2,1)
        if h2{i,3}
            h2{i,3} = h2{i,3} + n_h0 + n_h1;
        end
    end

    % make the boundary matrix
    boundary_matrix = zeros(n_h0+n_h1,...
                            n_h1+n_h2);
    
    all_h = cat(1,h0,h1,h2);

    % populate the boundary matrix
    for i=1:size(all_h,1)
        if all_h{i,3}
            points1 = all_h{i,2};
            for j=1:size(all_h,1)
                if all_h{j,3}
                    points2 = all_h{j,2};
                    if numel(points1) < numel(points2) && all(ismember(points1,points2))
                        enum1 = all_h{i,3};
                        enum2 = all_h{j,3};
                        boundary_matrix(enum1,enum2)=1;
                    end
                end
            end
        end
    end
    
    changed=1;
    
    while changed
        % reduce the boundary matrix
        changed = 0;
        for i=1:size(boundary_matrix,2) % for each column...
            one_locs_master = find(boundary_matrix(:,i)); % find the row of lowest 1
            if ~isempty(one_locs_master)
                lowest_master = one_locs_master(end);
            
                % check columns to the right if there's a match
                for j=i+1:size(boundary_matrix,2)
                    one_locs_slave = find(boundary_matrix(:,j));
                    if ~isempty(one_locs_slave)
                        lowest_slave = one_locs_slave(end);
                        if lowest_master == lowest_slave % apply column transformation
                            boundary_matrix(:,j) = double(xor(boundary_matrix(:,j),boundary_matrix(:,i)));
                            changed=1;
                        end
                    end
                end
            end
        end
    end
end

% read and preprocess data
function [time_vibration,vibration_signal,...
    time_pin,pin_position] = read_and_clean_dataset(filename,...
                                                    sample_rate,...
                                                    use_higher_sample_rate_for_inputs)
    % read data and compute sample rates
    data = jsondecode(fileread(filename));
    
    % native sample rates computed here
    %vibration_sample_rate = numel(data.acceleration_data) / (data.time_acceleration_data(end) - data.time_acceleration_data(1));
    %pin_sample_rate = numel(data.measured_pin_location) / (data.measured_pin_location_tt(end) - data.measured_pin_location_tt(1));
    
    % remove nans in pin position
    for i = find(isnan(data.measured_pin_location))
        data.measured_pin_location(i) = (data.measured_pin_location(i+1) + data.measured_pin_location(i-1))/2;
    end
    
    % determine overlapping time span for both signals
    latest_start_time = max([data.time_acceleration_data(1) data.measured_pin_location_tt(1)]);
    earliest_end_time = min([data.time_acceleration_data(end) data.measured_pin_location_tt(end)]);
    
    % trim signals
    clip_start = find(data.time_acceleration_data>=latest_start_time);
    clip_end = find(data.time_acceleration_data>=earliest_end_time);
    data.time_acceleration_data = data.time_acceleration_data(clip_start(1):clip_end(1));
    data.acceleration_data = data.acceleration_data(clip_start(1):clip_end(1));
    
    clip_start = find(data.measured_pin_location_tt>=latest_start_time);
    clip_end = find(data.measured_pin_location_tt>=earliest_end_time);
    data.measured_pin_location_tt = data.measured_pin_location_tt(clip_start(1):clip_end(1));
    data.measured_pin_location = data.measured_pin_location(clip_start(1):clip_end(1));
    
    % create new time axes
    if use_higher_sample_rate_for_inputs
        sample_rate_vib = sample_rate*number_of_sequence_inputs;
    else
        sample_rate_vib = sample_rate;
    end
    time_vibration = [data.time_acceleration_data(1):...
                        1/sample_rate_vib:...
                        data.time_acceleration_data(end)];
    
    time_pin = [data.measured_pin_location_tt(1):...
                        1/sample_rate:...
                        data.measured_pin_location_tt(end)];
    
    % interpolate signals
    vibration_signal = interp1(data.time_acceleration_data,data.acceleration_data,time_vibration);
    pin_position = interp1(data.measured_pin_location_tt,data.measured_pin_location,time_pin);
end

% read dataset
function [time_vibration,vibration_signal,...
        time_pin,pin_position,vibration_signal_samples,start_time] = read_data (sample_rate,dim,plotit,plottime)
    % read the input data
    [time_vibration,vibration_signal,...
        time_pin,pin_position] = read_and_clean_dataset('data_6_with_FFT.json', ...
                                                        sample_rate,...
                                                        0);
    
    % convert input data into moving window datapoints
    vibration_signal_samples = zeros(size(vibration_signal,2)-1,dim);
    for i=1:size(vibration_signal_samples,1)-1
        vibration_signal_samples(i,:) = vibration_signal(i:i+dim-1);
    end
    
    pts = find(time_vibration>plottime);
    pts = pts(1);

    if plotit
        % plot for sanity check
        figure;
        hold on;
        title('training data');
        plot(time_vibration(1:pts),vibration_signal(1:pts),'g');
        yyaxis right
        plot(time_pin(1:pts),pin_position(1:pts),'r');
        title('training data');
        %legend({"vibration","pin position"});
        xlabel('time (s)');
        %hold off;
        drawnow;
        yyaxis left
    end

    start_time = time_vibration(1);
end
