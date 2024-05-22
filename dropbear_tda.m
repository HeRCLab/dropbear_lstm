function [] = main ()

    %process_dropbear_data();
    %return

    data = [0.91322074, 0.93506856;...
           0.02695392, 0.34799628;...
           0.32018781, 0.9919    ;...
           0.76357508, 0.08647722;...
           0.92744649, 0.52848275];
           
    ripser_output = [0.        , 0.4068346 ;...
           0.        , 0.47140506;...
           0.        , 0.59574986;...
           0.        , 0.7075296 ;...
           0.        ,        inf;...
           0.78166676, 0.91840202];
    
    scatter(data(:,1),data(:,2));

    [distance_matrix,boundary_matrix1,boundary_matrix2,h0,h1,h2] = tda (data,1e38);
    persistance = get_persistances(boundary_matrix1,boundary_matrix2,h0,h1,h2);

    % extract features
    longest_distances1 = longest_persistance(distance_matrix,boundary_matrix1,h0,h1,1);
    longest_distances2 = longest_persistance(distance_matrix,boundary_matrix2,h1,h2,1);

    % % decode the ripser results
    % decoded = cell(size(ripser_output));
    % for i=1:size(ripser_output,1)
    %     for j=1:size(all_h,1)
    %         if ripser_output(i,1)>0 && abs(ripser_output(i,1)-all_h{j})<1e-5
    %             decoded{i,1}=all_h{j,2};
    %         end
    %         if ripser_output(i,2)>0 && abs(ripser_output(i,2)-all_h{j})<1e-5
    %             decoded{i,2}=all_h{j,2};
    %         end
    %     end
    % end

end
%     n_points=6;
%     n_homologies = n_points + (n_points^2-n_points)/2 + ...
%                 (n_points^3-3*n_points^2+2*n_points)/6;
%     for i = 1:n_homologies
%         find_points_from_enumeration (i,n_points)
%     end
%     return;

function [persistance] = get_persistances(boundary_matrix1,boundary_matrix2,h0,h1,h2)
    persistance = [];

    for i=1:size(boundary_matrix1,2)
        ones = find(boundary_matrix1(:,i)==1);
        if ~isempty(ones)
            birth = 0;%ones(end);
            death = h1{i,1};
            persistance = [persistance;birth,death];
        end
    end

    for i=1:size(boundary_matrix2,2)
        ones = find(boundary_matrix2(:,i)==1);
        if ~isempty(ones)
            birth = h1{ones(end),1};
            death = h2{i,1};
            if birth ~= death
                persistance = [persistance;birth,death];
            end
        end
    end
end

function [] = process_dropbear_data ()
    % set up parameters
    n = 1; % number of longest persistances

    sample_rate = 25600/32;
    dim = 2;
    use_higher_sample_rate_for_inputs = 0;
    radius = 1e6;

    tda_window = .04; % in seconds
    delay = .25 / 17.7;
    cutoff = 150;

    sim_time = 0.4;

    % read and plot data
    [time_vibration,vibration_signal,...
            time_pin,pin_position,vibration_signal_samples,start_time,sample_rate] = read_data (sample_rate,dim,1,5.5,delay,cutoff);
    
    % extract a window for TDA
    number_of_samples_in_window = sample_rate * tda_window;
    
    % only compute TDA features for the first window
    % note this will eventually be a loop to compute all windows
    n_windows = (size(vibration_signal_samples,1)-number_of_samples_in_window+1);

    % overridden to perform a partial run
    n_windows = floor(sim_time * sample_rate);

    longest_distances = zeros(n_windows,n);
    for window_number = 1:n_windows
        %sample_range_in_window = (window_number-1)*number_of_samples_in_window+1:window_number*number_of_samples_in_window;
        sample_range_in_window = window_number:window_number+number_of_samples_in_window-1;
        window_data = vibration_signal_samples(sample_range_in_window,:);

        % perform TDA
        [distance_matrix,boundary_matrix1,boundary_matrix2,h0,h1,h2] = tda (window_data,radius);

        % extract features
        longest_distances1 = longest_persistance(distance_matrix,boundary_matrix1,h0,h1,1);
        longest_distances2 = longest_persistance(distance_matrix,boundary_matrix2,h1,h2,1);

        % merge
        n1=1;
        n2=1;
        n=1;
        while n1<=numel(longest_distances1) && n2<=numel(longest_distances2)
            if longest_distances1(n1) > longest_distances1(n2)
                longest_distances(window_number,n) = longest_distances1(n1);
                n1=n1+1;
            else
                longest_distances(window_number,n) = longest_distances1(n2);
                n2=n2+1;
            end
            n = n+1;
        end
        for i=n1:numel(longest_distances1)
            longest_distances(window_number,n) = longest_distances1(i);
            n=n+1;
        end
        for i=n2:numel(longest_distances1)
            longest_distances(window_number,n) = longest_distances2(i);
            n=n+1;
        end

        if mod(window_number,100)
            plot((1:window_number)./sample_rate+start_time,longest_distances(1:window_number),'m','Marker','None');
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

function longest_distances = longest_persistance(distance_matrix,boundary_matrix,h1,h2,n)

    longest_distances = zeros(1,n);
    
    for i=1:size(boundary_matrix,2)
        col = boundary_matrix(:,i);
        idx = find(col);
        start = 0;
        if ~isempty(idx)
            start=idx(end);
        end

         tmp=find(boundary_matrix(:,i));
        if ~isempty(tmp)
            % get enum ID of birth
            birth_idx = tmp(end);
            % get index in the distance table of death enum
            tmp=find(cell2mat(h2(:,3))==i);
            idx_in_distance_table = tmp(end);
            % lookup death time
            death_time = cell2mat(h2(idx_in_distance_table,1));

            % get index in the distance table of birth enum
            tmp=find(cell2mat(h1(:,3))==birth_idx);
            idx_in_distance_table = tmp(end);

            % lookup birth time
            birth_time = cell2mat(h1(idx_in_distance_table,1));

            persistance = death_time-birth_time;
        end

        for j=1:n
            if persistance > longest_distances(j)
                longest_distances = [longest_distances(1,1:j-1),persistance,longest_distances(1,j:n-1)];
                break;
            end
        end
    end

end

% find the death time of a given homology
% assume a time horizon of the largest distance
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
            h1{cnt,3} = cnt;
            cnt=cnt+1;
        end
    end

    cnt=1;
    for i=1:n-2
        for j=i+1:n-1
            for k=j+1:n
                h2{cnt,1} = max([dist_matrix(i,j),dist_matrix(i,k),dist_matrix(j,k)]);
                h2{cnt,2} = [i,j,k];
                h2{cnt,3} = cnt;
                cnt=cnt+1;
            end
        end
    end

end

function [h0,h1,h2] = create_back_refs(h0,h1,h2)
    % back-reference to h1
    for i=1:size(h1,1)
        h1{i,4} = h1{i,2};
    end

    for i=1:size(h2,1)
        members=h2{i,2};
        back_refs=[0,0,0];
        cnt=1;
        for j=1:size(h1,1)
            subset=h1{j,2};
            if all(ismember(subset,members))
                back_refs(cnt)=j;
                cnt=cnt+1;
                if cnt==4
                    break;
                end
            end
        end
        h2{i,4} = back_refs;
    end
end

function [boundary_matrix] = generate_boundary (h1,h2)
    % allocate the boundary matrix
    boundary_matrix = zeros(size(h1,1),size(h2,1));

    for i=1:size(h2,1)
        for j=h2{i,4}
            boundary_matrix(j,i)=1;
        end
    end

    return

    % populate the boundary matrix
    for i=1:size(h1,1)
        if h1{i,3}
            points1 = h1{i,2};
            for j=1:size(h2,1)
                if h2{j,3}
                    points2 = h2{j,2};
                    if all(ismember(points1,points2))
                        enum1 = h1{i,3};
                        enum2 = h2{j,3};
                        boundary_matrix(enum1,enum2)=1;
                    end
                end
            end
        end
    end
end

function boundary_matrix = reduce_boundary(boundary_matrix)
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

% return boundary matrix and distance matrix for input data
function [distance_matrix,...
    boundary_matrix1_reduced,...
    boundary_matrix2_reduced,...
    h0,h1,h2] = tda (window_data,radius)

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

    [h0,h1,h2]=create_back_refs(h0,h1,h2);

    n_h0 = size(distance_matrix,1);

    n_h1 = size(h1,1);
    n_h2 = size(h2,1);

    boundary_matrix1 = generate_boundary (h0,h1);
    boundary_matrix2 = generate_boundary (h1,h2);
    
    boundary_matrix1_reduced = reduce_boundary (boundary_matrix1);
    boundary_matrix2_reduced = reduce_boundary (boundary_matrix2);

end

% read and preprocess data
function [time_vibration,vibration_signal,...
    time_pin,pin_position,sample_rate] = read_and_clean_dataset(filename,...
                                                    sample_rate,...
                                                    use_higher_sample_rate_for_inputs)
    % read data and compute sample rates
    data = jsondecode(fileread(filename));
    
    if sample_rate==0
        sample_rate = data.accelerometer_sample_rate;
    end

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
    
    % interpolate signals (if needed)
    vibration_signal = interp1(data.time_acceleration_data,data.acceleration_data,time_vibration);
    pin_position = interp1(data.measured_pin_location_tt,data.measured_pin_location,time_vibration);

end

% read dataset
function [time_vibration,vibration_signal,...
        time_pin,pin_position,vibration_signal_samples,start_time,sample_rate] = read_data (sample_rate,dim,plotit,plottime,delay,cutoff)

    % read the input data
    [time_vibration,vibration_signal,...
        time_pin,pin_position,sample_rate] = read_and_clean_dataset('data_6_with_FFT.json', ...
                                                        sample_rate,...
                                                        0);
    
    % apply FLP
    if cutoff ~= 0
        vibration_signal = lowpass(vibration_signal,cutoff,sample_rate);
    end

    % TODO: normalize data and apply LPF and allow for no subsampling
    % and allow for window stride
    
    if delay ~= 0
        sample_rate = numel(time_pin)/(time_pin(end) - time_pin(1));
        stride = floor(sample_rate * delay);
    else
        stride = 1;
    end

    % convert input data into moving window datapoints
    vibration_signal_samples = zeros(size(vibration_signal,2)-1,dim);
    for i=1:size(vibration_signal_samples,1)-stride*dim-1
        vibration_signal_samples(i,:) = vibration_signal(i:stride:i+stride*dim-1);
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
