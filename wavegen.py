# Sine wave generator/mixer.
import json
import datetime
import argparse
import numpy as np
from scipy import signal as sg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_rate", type=int, help="Sampling rate. (Hz)")
    parser.add_argument("num_samples", type=int, help="Number of samples to generate.")
    parser.add_argument("freqs", nargs='+', type=int, help="Frequencies to generate. (Hz)")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot using matplotlib.")

    args = parser.parse_args()
    #print(args)  # DEBUG

    sampling_rate = args.sample_rate
    num_samples = args.num_samples
    freqs = args.freqs

    x = np.arange(num_samples) # x-axis for the plots.

    # Build a dictionary of arrays of the form:
    # {frequency: [x1, x2, x3, x4, ..., xn]}
    y = {}
    for freq in freqs:
        y[freq] = 100 * np.sin(2 * np.pi * freq * x / sampling_rate)

    # Create combined signal.
    combined = np.zeros(num_samples)
    for freq in freqs:
        combined += y[freq]

    # Dump data as JSON to stdout.
    out = {
        "date_time": str(datetime.datetime.now(datetime.timezone.utc)),
        "acceleration_data": list(combined),
        "accelerometer_sample_rate": sampling_rate,
        "time_accelerometer_aquire_start": 0.0,
        "time_acceleration_data": [float(y) for y in x],
    }
    print(json.dumps(out))

    # Plot only if the user asked for plotting.
    if args.plot:
        import matplotlib.pyplot as plt  # Intentionally late import.

        # Plot subplot for each signal.
        fig, axs = plt.subplots(len(freqs)+1, 1, sharex=True)
        for (i, freq) in zip(range(len(freqs)+1), freqs):
            axs[i].plot(x, y[freq], '-')
            axs[i].set_ylabel('{} Hz'.format(freq))

        # Plot combined signal.
        axs[len(freqs)].plot(x, combined, '-')
        axs[len(freqs)].set_ylabel('Combined'.format(freq))

        # Put a title on the plot and the window, then render.
        fig.suptitle('Combining signals at different frequencies', fontsize=15)
        fig.canvas.set_window_title('Signals')
        plt.show()
