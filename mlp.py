#!/usr/bin/python3
# MLP/ANN script. Intended for use with data from the wavegen script.
import sys
import math
import json
import collections
import argparse
from datetime import datetime
import numpy as np
import scipy.signal as signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clear_session
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='ANN/MLP Model')
    parser.add_argument('--online', action="store_true", help="Do online training using a single model. (default: False)")  # noqa: E501
    parser.add_argument('-a', '--activation', action="append", default=[], help="Activation function for the hidden layer(s). Providing this argument more than once will set the activation function for successive hidden layers. (default: linear)")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to spend training the network. (default: 10)")  # noqa: E501
    parser.add_argument('-f', '--forecast-length', type=int, default=1, help="Number of samples to predict at a time (default: 1)")  # noqa: E501
    parser.add_argument('-g', '--prediction-gap', type=int, default=0, help="Number of samples after window, before predicted region starts (default: 0)")  # noqa: E501
    parser.add_argument('-u', '--units', action="append", default=[], help="Number of units in hidden layer. Providing this argument more than once will add additional hidden layers. (default: 1)")  # noqa: E501
    parser.add_argument('-s', '--sampling-window', type=int, default=10, help="Number of samples to slice from the window for each training example. (default: 10)")  # noqa: E501
    parser.add_argument('-t', '--training-window', type=int, default=40, help="Number of samples in each training window. (default: 40)")  # noqa: E501
    parser.add_argument('-i', '--filename', type=str, help="Filename of JSON waveform data to read in.")  # noqa: E501
    parser.add_argument('-p', '--plot', action="store_true", help="Plot using matplotlib. (default: False)")  # noqa: E501
    parser.add_argument('-v', '--verbose', action="store_true", help="Show more debugging information. (default: False)")  # noqa: E501
    parser.add_argument('--show-rmse-per-window', action="store_true", help="Display RMSE values for each training window. (default: False)")  # noqa: E501
    parser.add_argument('--use-gpu', action="store_true", help="Use GPU for accelerating model training. (default: False)")  # noqa: E501

    return parser.parse_args()


def generate_training_samples(x,
                              training_window,
                              sampling_window,
                              prediction_gap,
                              forecast_length):
    assert training_window >= sampling_window, "Sampling window must be smaller than training window."
    end_train = len(x)-prediction_gap-forecast_length-training_window
    out = []

    for i in range(0, end_train, training_window):
        x_train = []
        y_train = []

        # Prepare training data.
        for j in range(0, training_window):
            start_idx = i - sampling_window + j
            if start_idx < 1:
                continue
            end_idx = i + j
            # Slice off `training_window` of input data.
            x_train += [x[start_idx:end_idx]]
            # Slice off biiiiig
            end_idx = end_idx+prediction_gap
            y_train += [x[end_idx:end_idx+forecast_length]]
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        out.append((i, x_train, y_train))

    return out


def create_model(units, activations, sampling_window, output_layer_size=1, want_verbose=0):
    model = Sequential()
    top_layer = True
    activations = list(activations) # Clone the list so it can be consumed.
    # If provided a list, it will create 1 layer per integer list item.
    if isinstance(units, collections.Iterable):
        for u in units:
            if top_layer:
                model.add(Dense(u, input_shape=(sampling_window,), activation=activations.pop(0)))
                top_layer = False
            else:
                model.add(Dense(u, activation=activations.pop(0)))
    # Otherwise, it will just make a single hidden layer.
    else:
        model.add(Dense(units, input_shape=(sampling_window,), activation=activations[0]))
    model.add(Dense(output_layer_size))  # Default: 1
    model.compile(loss='mean_squared_error', optimizer='adam', verbose=want_verbose)
    return model


def train_model(units,
                activations,
                x_train,
                y_train,
                sampling_window,
                forecast_length,
                model=None,
                epochs=10,
                want_gpu=False,
                want_verbose=0):
    # the following line verifies that Tensorflow has detected a target GPU
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Windowed-style
    if model is None:
        if want_gpu:
            with tf.device('/gpu:0'):
                model = create_model(units, activations, sampling_window, forecast_length)
                model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)  # noqa: E501
                # Reset states for LSTMs here.
        else:
            model = create_model(units, activations, sampling_window, forecast_length)
            model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)
    # Online-style
    else:
        model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)

    return model

# Predicts a future window of data, in chunks of forecast_length.
def iterative_predict_window(model, x_train, training_window, forecast_length):
    out_window = np.zeros((training_window,))
    for i in range(0, training_window, forecast_length):
        future_x = np.array([x_train[i]])
        y_pred = model.predict([future_x])
        # print("Prediction: {}".format(y_pred))
        y_pred = y_pred.flatten()
        # Do some funky slicing on the last window, leave others alone.
        end_idx = i+len(y_pred)
        y_pred = [y_pred, y_pred[:end_idx-training_window]][end_idx > training_window]
        out_window[i:min(i+len(y_pred), training_window)] = y_pred
    return out_window


def main():
    args = parse_args()
    # downsample_levels, sampling_windows, time_shift, training_portion = load_config (args.config)  # noqa: E501

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
    activations = args.activation
    # Ensure every hidden layer gets an explicit activation function.
    if len(units) > len(activations):
        len_diff = len(units) - len(activations)
        activations += ["linear" for x in range(0, len_diff)]
    prediction_gap = args.prediction_gap
    forecast_length = args.forecast_length
    training_window = args.training_window
    sampling_window = args.sampling_window
    model = None
    prev_model = None
    use_gpu = args.use_gpu
    want_verbose = args.verbose

    y_predicted = np.zeros((len(x),))
    results_idx = training_window

    # Online training fun
    training_examples = generate_training_samples(x, training_window, sampling_window, prediction_gap, forecast_length)

    for (idx, x_train, y_train) in training_examples:
        # PREDICTION (Normal)
        # Prediction across this buffer of samples, using the previous
        # window's model.
        if prev_model is not None:
            y_pred = iterative_predict_window(prev_model, x_train, training_window, forecast_length)
            y_predicted[results_idx:results_idx+len(y_pred)] = y_pred
            results_idx += len(y_pred)

        # Normalize the training data.
        # TODO

        # Train the model + get predictions.
        if args.online and model is not None:
            model = train_model(units, activations, x_train, y_train, sampling_window, forecast_length, epochs=epochs, want_gpu=use_gpu, model=model, want_verbose=want_verbose)  # noqa: E501
        else:
            model = train_model(units, activations, x_train, y_train, sampling_window, forecast_length, epochs=epochs, want_gpu=use_gpu, want_verbose=want_verbose)  # noqa: E501
        y_pred = model.predict(x_train)

        prev_model = model  # Swap in the last window's model.

        # Clean up all excess state Keras is keeping around.
        clear_session()

    # Computed global RMSE.
    size = len(y_predicted)-(training_window*2)
    diffs = np.ndarray((size,), float)
    for i in range(0, size):
        diffs[i] = np.square(y_predicted[i+training_window] - x[i+training_window])  # noqa: E501
    rmse = math.sqrt(np.mean(diffs))
    # print("Global RMSE: {}".format(rmse))

    # Print JSON blob for other tools to consume.
    out = {
        "author": "Philip Conrad",
        "algorithm": ["window-mlp", "window-mlp-online"][args.online],
        "activation": "linear",
        "dataset": ["unknown", args.filename][args.filename is not None],
        "creation_ts": datetime.utcnow().isoformat(),
        "forecast_length": forecast_length,
        "prediction_gap": prediction_gap,
        "training_window_length": training_window,
        "sample_window_length": sampling_window,
        "epochs": epochs,
        "layers": "-".join([str(x) for x in units]),
        "rmse_global": rmse,
        "metadata": json.dumps({}),
    }
    print(json.dumps(out))

    # Plot only if the user asked for plotting.
    if args.plot:
        import matplotlib.pyplot as plt  # Intentionally late import.

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
