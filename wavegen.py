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
    parser.add_argument("-m", "--magnitude", type=int, default=1, help="Magnitude for all signals. (default: 1)")
    parser.add_argument("-n", "--noise", default="uniform", type=str, help="Type of noise to add. (default: uniform)")
    parser.add_argument("-nm", "--noise-magnitude", type=float, default=0.0, help="Magnitude of uniform white noise to mix in. (default: 0.0)")


    args = parser.parse_args()
    #print(args)  # DEBUG

    sampling_rate = args.sample_rate
    num_samples = args.num_samples
    freqs = args.freqs
    magnitude = args.magnitude
    noise_type = args.noise
    noise_magnitude = args.noise_magnitude

    x = np.arange(num_samples) # x-axis for the plots.
    noise = np.zeros(num_samples)
    if noise_type == "uniform":
        noise = np.random.uniform(-noise_magnitude, noise_magnitude, num_samples) # noise buffer.
    elif noise_type == "normal":
        noise = np.random.normal(0, 0.1, num_samples) # mu, sigma, num_samples
    else:
        print("Unknown noise type: '{}'".format(noise_type))
        exit(1)

    # Build a dictionary of arrays of the form:
    # {frequency: [x1, x2, x3, x4, ..., xn]}
    y = {}
    for freq in freqs:
        y[freq] = magnitude * np.sin(2 * np.pi * freq * x / sampling_rate)

    # Create combined signal.
    combined = np.zeros(num_samples)
    for freq in freqs:
        combined += y[freq]
    # Add in the noise.
    if noise_magnitude != 0.0:
        combined += noise

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

        num_plots = len(freqs)+1
        if noise_magnitude != 0.0:
            num_plots += 1

        # Plot subplot for each signal.
        fig, axs = plt.subplots(num_plots, 1, sharex=True)
        for (i, freq) in zip(range(num_plots), freqs):
            axs[i].plot(x, y[freq], '-')
            axs[i].set_ylabel('{} Hz'.format(freq))

        # Plot noise buffer, if enabled.
        if noise_magnitude != 0.0:
            axs[num_plots-2].plot(x, noise, '-')
            axs[num_plots-2].set_ylim(-magnitude, magnitude) # Scale to -1.0..1.0 range.
            axs[num_plots-2].set_ylabel('Noise')

        # Plot combined signal.
        axs[num_plots-1].plot(x, combined, '-')
        axs[num_plots-1].set_ylabel('Combined'.format(freq))

        # Put a title on the plot and the window, then render.
        fig.suptitle('Combining signals at different frequencies', fontsize=15)
        fig.canvas.set_window_title('Signals')
        plt.show()
