# Sine wave generator/mixer.
import json
import datetime
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt

# Wave properties.
sampling_rate = 16000 # Sample rate
freqs = [1, 3, 5, 10] # Frequency (Hz)
num_samples = 16000

# Other globals.
x = np.arange(num_samples) # x-axis for the plots.


if __name__ == '__main__':
    # Build a dictionary of arrays of the form:
    # {frequency: [x1, x2, x3, x4, ..., xn]}
    y = {}
    for freq in freqs:
        y[freq] = 100 * np.sin(2 * np.pi * freq * x / sampling_rate)

    # Plot subplot for each signal.
    fig, axs = plt.subplots(len(freqs)+1, 1, sharex=True)
    for (i, freq) in zip(range(len(freqs)+1), freqs):
        axs[i].plot(x, y[freq], '-')
        axs[i].set_ylabel('{} Hz'.format(freq))

    # Create combined signal.
    combined = np.zeros(num_samples)
    for freq in freqs:
        combined += y[freq]

    # Plot combined signal.
    axs[len(freqs)].plot(x, combined, '-')
    axs[len(freqs)].set_ylabel('Combined'.format(freq))

    # Put a title on the plot and the window, then render.
    fig.suptitle('Combining signals at different frequencies', fontsize=15)
    fig.canvas.set_window_title('Signals')
    plt.show()

    # Export data as JSON.
    out = {
        "date_time": str(datetime.datetime.now(datetime.timezone.utc)),
        "acceleration_data": list(combined),
        "accelerometer_sample_rate": sampling_rate,
        "time_accelerometer_aquire_start": 0.0,
        "time_acceleration_data": [float(y) for y in x],
    }
    print(json.dumps(out))
