import json
import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt

# Set up parameters
sample_rate = 10000
dim = 2
use_higher_sample_rate_for_inputs = 0
tda_window = 0.01
number_of_sequence_inputs = 1

# Define the 
def read_and_clean_dataset(filename, sample_rate, use_higher_sample_rate_for_inputs):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    for i in range(len(data['measured_pin_location'])):
        if np.isnan(data['measured_pin_location'][i]):
            data['measured_pin_location'][i] = (data['measured_pin_location'][i-1] + data['measured_pin_location'][i+1]) / 2

    # Determine overlapping time span for both signals
    latest_start_time = max(data['time_acceleration_data'][0], data['measured_pin_location_tt'][0])
    earliest_end_time = min(data['time_acceleration_data'][-1], data['measured_pin_location_tt'][-1])

    # Trim signals
    clip_start = np.where(np.array(data['time_acceleration_data']) >= latest_start_time)[0][0]
    clip_end = np.where(np.array(data['time_acceleration_data']) >= earliest_end_time)[0][0]
    data['time_acceleration_data'] = data['time_acceleration_data'][clip_start:clip_end]
    data['acceleration_data'] = data['acceleration_data'][clip_start:clip_end]

    clip_start = np.where(np.array(data['measured_pin_location_tt']) >= latest_start_time)[0][0]
    clip_end = np.where(np.array(data['measured_pin_location_tt']) >= earliest_end_time)[0][0]
    data['measured_pin_location_tt'] = data['measured_pin_location_tt'][clip_start:clip_end]
    data['measured_pin_location'] = data['measured_pin_location'][clip_start:clip_end]

    # Create new time axes
    if use_higher_sample_rate_for_inputs:
        sample_rate_vib = sample_rate * number_of_sequence_inputs
    else:
        sample_rate_vib = sample_rate

    time_vibration = np.arange(data['time_acceleration_data'][0], data['time_acceleration_data'][-1], 1/sample_rate_vib)
    time_pin = np.arange(data['measured_pin_location_tt'][0], data['measured_pin_location_tt'][-1], 1/sample_rate)

    # Interpolate signals
    vibration_signal = np.interp(time_vibration, data['time_acceleration_data'], data['acceleration_data'])
    pin_position = np.interp(time_pin, data['measured_pin_location_tt'], data['measured_pin_location'])

    return time_vibration, vibration_signal, time_pin, pin_position

time_vibration, vibration_signal, time_pin, pin_position = read_and_clean_dataset('data_6_with_FFT.json', sample_rate, use_higher_sample_rate_for_inputs)



# Convert input data into moving window datapoints
vibration_signal_samples = np.zeros((len(vibration_signal) - 1, dim))
for i in range(len(vibration_signal_samples) - 1):
    vibration_signal_samples[i, :] = vibration_signal[i:i+dim]
    
    
radius = 1e5
tda_window = 0.01  # in seconds
sim_time = 4
dim = 1

# Extract a window for TDA
number_of_samples_in_window = int(sample_rate * tda_window)

# Only compute TDA features for the first window
n_windows = len(vibration_signal_samples) - number_of_samples_in_window + 1

# Overridden to perform a partial run
n_windows = int(sim_time * sample_rate)

# Compute TDA features
longest_distances = np.zeros((n_windows, 1))
for window_number in range(n_windows):
    sample_range_in_window = slice(window_number, window_number + number_of_samples_in_window)
    window_data = vibration_signal_samples[sample_range_in_window, :]

    # Perform TDA
    result = ripser(window_data, maxdim=dim, thresh=radius)
    diagrams = result['dgms']
    
    if len(diagrams[1]) > 0:  # Check if diagrams[1] is not empty
       # Extract features
       longest_distances[window_number] = np.max(diagrams[1][:, 1])  # Using H1 persistence
    else:
       longest_distances[window_number] = 0  # Set to default value or handle appropriately

    # Plot every 100 windows
    if window_number % 100 == 0:
        plt.title('Longest Persistence and Pin Position')
        plt.plot(np.arange(window_number + 1) / sample_rate, longest_distances[:window_number + 1], 'm')
        plt.xlabel('Time (s)')
        plt.ylabel('Longest Persistence')
        plt.show()

# Plot pin position over time
# =============================================================================
# plt.figure()
# plt.plot(time_pin[:n_windows], pin_position[:n_windows])
# plt.xlabel('Time (s)')
# plt.ylabel('Displacement (m)')
# plt.show()
# =============================================================================

# Plot pin position and longest persistence over time on the same graph
fig, ax1 = plt.subplots()

# Plot pin position, shifted to start at 0
ax1.plot(time_pin[:n_windows]-1, pin_position[:n_windows], color='b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pin Position (m)', color='b')

# Create a secondary y-axis for longest persistence
ax2 = ax1.twinx()
ax2.plot(np.arange(n_windows) / sample_rate, longest_distances[:n_windows], color='r')
ax2.set_ylabel('Longest Persistence', color='r')

plt.show()