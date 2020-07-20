#!/usr/bin/python3
# MLP/ANN script. Intended for use with data from the wavegen script.
import os
import sys
import math
import json
import collections
import argparse
from datetime import datetime
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Input
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt
import tensorflow as tf     # issue with importing tensorflow on
                            # tachyon due to its lack of AVX instructions
                            # NOTE:  must use on higgs for GPU


def parse_args():
    parser = argparse.ArgumentParser(description='ANN/MLP Model')
    parser.add_argument('--online', action="store_true", help="Do online training using a single model. (default: False)")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to spend training the network. (default: 10)")
    parser.add_argument('-u', '--units', action="append", default=[], help="Number of units in hidden layer. Providing this argument more than once will add additional hidden layers. (default: 1)")
    parser.add_argument('-s', '--sampling-window', type=int, default=10, help="Number of samples to slice from the window for each training example. (default: 10)")
    parser.add_argument('-t', '--training-window', type=int, default=40, help="Number of samples in each training window. (default: 40)")
    parser.add_argument('-f', '--filename', type=str, help="Filename of JSON waveform data to read in.")
    parser.add_argument('-p', '--plot', action="store_true", help="Plot using matplotlib. (default: False)")
    parser.add_argument('-v', '--verbose', action="store_true", help="Show more debugging information. (default: False)")
    parser.add_argument('--show-rmse-per-window', action="store_true", help="Display RMSE values for each training window. (default: False)")
    parser.add_argument('--use-gpu', action="store_true", help="Use GPU for accelerating model training. (default: False)")

    return parser.parse_args()


def create_model(units, history_length, want_verbose=0):
    model = Sequential()
    top_layer = True
    # If provided a list, it will create 1 layer per integer list item.
    if isinstance(units, collections.Iterable):
        for u in units:
            if top_layer:
                model.add(Dense(u, input_shape=(history_length,)))
                top_layer = False
            else:
                model.add(Dense(u))
    # Otherwise, it will just make a single hidden layer.
    else:
        model.add(Dense(units, input_shape=(history_length,)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', verbose=want_verbose)
    return model


def train_model(units, x_train, y_train, history_length, model=None, epochs=10, want_gpu=False, want_verbose=0):
    # the following line verifies that Tensorflow has detected a target GPU
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Windowed-style
    if model is None:
        if want_gpu:
            with tf.device('/gpu:0'):
                model = create_model(units, history_length)
                model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)
                #model.reset_states() # Only needed for stateful LSTM.
        else:
            model = create_model(units, history_length)
            model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)
    # Online-style
    else:
        model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)

    return model


def main():
    args = parse_args()
    #downsample_levels, history_lengths, time_shift, training_portion = load_config (args.config)

    # Read contents of file (if provided), else read from stdin.
    data = None
    if args.filename is not None:
        with open(args.filename, "r") as f:
            data = json.loads(f.read()) 
    else:
        text = "".join([line for line in sys.stdin])
        data = json.loads(text)

    x = np.array(data["acceleration_data"])
    input_sample_rate = np.float(data["accelerometer_sample_rate"])

    epochs = args.epochs
    if len(args.units) > 0:
        units = [int(x) for x in args.units]
    else:
        units = [1]
    prediction_time = 1
    training_window = args.training_window
    history_length = args.sampling_window
    model = None
    prev_model = None
    use_gpu = args.use_gpu
    want_verbose = args.verbose

    y_predicted = np.zeros((len(x),))
    results_idx = training_window

    # Online training fun
    for i in range(0, len(x)-prediction_time-training_window, training_window):
        start_train = i
        end_train = i + training_window
        x_train = []
        y_train = []

        # Prepare training data.
        for j in range(0, training_window):
            start_idx = i - history_length + j
            if start_idx < 1:
                continue
            end_idx = i + j
            x_train += [x[start_idx:end_idx]]
            y_train += [x[end_idx+prediction_time]]
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # PREDICTION (Normal)
        # Prediction across this buffer of samples, using the previous window's model.
        if prev_model is not None:
            for i in range(0, training_window):
                future_x = np.array([x_train[i]])
                y_pred = prev_model.predict([future_x])
                #print("Prediction: {}".format(y_pred))
                y_pred = y_pred.flatten()
                y_predicted[results_idx:results_idx+len(y_pred)] = y_pred
                results_idx += len(y_pred)

        # Normalize the training data.
        # TODO

        # Train the model + get predictions.
        if args.online and model is not None:
            model = train_model(units, x_train, y_train, history_length, epochs=epochs, want_gpu=use_gpu, model=model)
        else:
            model = train_model(units, x_train, y_train, history_length, epochs=epochs, want_gpu=use_gpu)
        y_pred = model.predict(x_train)

        prev_model = model # Swap in the last window's model.

        # Clean up all excess state Keras is keeping around.
        clear_session()

    # Computed global RMSE.
    size = len(y_predicted)-(training_window*2)
    diffs = np.ndarray((size,), float)
    for i in range(0, size):
        diffs[i] = np.square(y_predicted[i+training_window] - x[i+training_window])
    rmse = math.sqrt(np.mean(diffs))
    #print("Global RMSE: {}".format(rmse))

    # Print JSON blob for other tools to consume.
    out = {
        "author": "Philip Conrad",
        "algorithm": ["window-mlp", "window-mlp-online"][args.online],
        "activation": "linear",
        "dataset": ["unknown", args.filename][args.filename is not None],
        "creation_ts": datetime.utcnow().isoformat(),
        "forecast_length": 1,
        "prediction_gap": 0,
        "training_window_length": training_window,
        "sample_window_length": history_length,
        "epochs": epochs,
        "layers": "-".join([str(x) for x in units]),
        "rmse_global": rmse,
        "metadata": json.dumps({}),
    }
    print(json.dumps(out))

    # Plot only if the user asked for plotting.
    if args.plot:
        import matplotlib.pyplot as plt  # Intentionally late import.

        #num_plots = 3
        num_plots = 2
        x_data = np.arange(len(x))

        # Plot subplot for each signal.
        fig, axs = plt.subplots(num_plots, 1, sharex=True)
        y_data = x
        axs[0].plot(x_data, y_data, '-')
        axs[0].set_ylabel('Signal')

        y_data = y_predicted
        axs[1].plot(x_data, y_data, 'r-')
        axs[1].plot(x_data, x, 'b-')
        axs[1].set_ylabel('Predicted (w/real data)')

        # Put a title on the plot and the window, then render.
        fig.suptitle('(MLP) Original vs Predicted signals', fontsize=15)
        fig.canvas.set_window_title('MLP Predicted Signals')
        plt.show()


if __name__ == '__main__':
    main()
