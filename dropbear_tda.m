tic

% set up parameters
sample_rate = 1000;
dim = 2;
use_higher_sample_rate_for_inputs = 0;

tda_window = .01; % in seconds

% read the input data
[time_vibration,vibration_signal,...
    time_pin,pin_position] = read_and_clean_dataset('data_6_with_FFT.json', ...
                                                    sample_rate,...
                                                    use_higher_sample_rate_for_inputs);

% convert input data into moving window datapoints
vibration_signal_samples = zeros(size(vibration_signal,2)-1,dim);
for i=1:size(vibration_signal_samples,1)-1
    vibration_signal_samples(i,:) = vibration_signal(i:i+dim-1);
end

% extract window for TDA
number_of_samples_in_window = sample_rate * tda_window;

% only compute TDA features for the first window
% note this will eventually be a loop to compute all windows
window_number = 1;
sample_range_in_window = (window_number-1)*number_of_samples_in_window+1:window_number*number_of_samples_in_window;
window_data = vibration_signal_samples(sample_range_in_window,:);

% fill in distance matrix
distance_matrix = zeros(size(window_data,1),size(window_data,1));
for i=1:size(window_data,1)
    for j=1:size(window_data,1)
        distance_matrix(i,j) = norm(window_data(i,:)-window_data(j,:));
    end
end

boundary_list = {};

% find the earliest birth of H-2's
for i=1:size(distance_matrix,1)-1
    [dist,j]=min(distance_matrix(i,i+1:end));
    boundary_list = [boundary_list;{[i j],dist}];
end

% find the earliest birth of H-3's
for i=1:size(distance_matrix,1)
    for j=i+1:size(distance_matrix,1)
        % extract the distances from node i to all others
        dist1 = distance_matrix(i,i+1:end);
        dist_idx = i+1:size(distance_matrix,2);
        %  extract the distances from node j to all others
        dist2 = distance_matrix(j,i+1:end);
        % remove node j
        dist1(j-i)=[];
        dist2(j-i)=[];
        dist_idx(j-i)=[];

        % find the minimum maximum distance from nodes i and j
        dist_i_j = ones(1,numel(dist1))*distance_matrix(i,j);
        [dist,k_idx]=min(max([dist1;dist2;dist_i_j]));

        % add to boundary list
        boundary_list = [boundary_list;{[i j dist_idx(k_idx)],dist}];
    end
end

% make the boundary matrix
boundary_matrix = zeros(size(boundary_list,1),size(distance_matrix,1));

% populate the boundry matrx
for i=1:size(boundary_list,1)
    points = boundary_list{i,1};
    for j=1:size(points,2)
        boundary_matrix(i,points(j))=1;
    end
end

toc

idx = 1:size(distance_matrix,1);
idx = repmat(idx,1,size(distance_matrix,1));
idx = idx';

idx2 = 1:size(distance_matrix,1);
idx2 = repmat(idx2,size(distance_matrix,1),1);
idx2 = idx2(:);

sorted_distances = [idx idx2 distance_matrix(:)];
sorted_distances = sortrows(sorted_distances,3);

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
