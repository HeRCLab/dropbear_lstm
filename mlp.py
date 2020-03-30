#!/usr/bin/python3
# MLP/ANN script. Intended for use with data from the wavegen script.
import os
import sys
import math
import json
import collections
import argparse
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Input
import matplotlib.pyplot as plt
import tensorflow as tf     # issue with importing tensorflow on
                            # tachyon due to its lack of AVX instructions
                            # NOTE:  must use on higgs for GPU


def parse_args():
    parser = argparse.ArgumentParser(description='ANN/MLP Model')
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to spend training the network. (default: 10)")
    parser.add_argument('-u', '--units', action="append", default=[1], help="Number of units in hidden layer. Providing this argument more than once will add additional hidden layers. (default: 1)")
    parser.add_argument('--history', '--history-length', type=int, default=10, help="Number of samples in the history buffer. (default: 10)")
    parser.add_argument('-w', '--training-window', type=int, default=40, help="Number of samples in each training window. (default: 40)")
    parser.add_argument('-f', '--filename', type=str, help="Filename of JSON waveform data to read in.")
    parser.add_argument('-t', '--training-ratio', type=float, help="Ratio of incoming data to use for training, on a scale of 0.0-1.0 (default: 0.1)")
    parser.add_argument('-p', '--plot', action="store_true", help="Plot using matplotlib. (default: False)")
    parser.add_argument('--show-rmse-per-window', action="store_true", help="Display RMSE values for each training window. (default: False)")
    parser.add_argument('--use-gpu', action="store_true", help="Use GPU for accelerating model training. (default: False)")

    return parser.parse_args()


def create_model(units, history_length):
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
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(units, x_train, y_train, history_length, epochs=10, want_gpu=False):
    # the following line verifies that Tensorflow has detected a target GPU
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    #batchsize = x_train.shape[0]
    #print("BATCH SIZE: {}".format(batchsize))
    #batchsize = 1

    if want_gpu:
        with tf.device('/gpu:0'): 
            model = create_model(units, history_length)
            model.fit(x_train, y_train, epochs=epochs, verbose=0)
            #model.reset_states() # Only needed for stateful LSTM.
    else:
        model = create_model(units, history_length)
        model.fit(x_train, y_train, epochs=epochs, verbose=0)

    return model


def main():
    args = parse_args()
    #downsample_levels, history_lengths, time_shift, training_portion = load_config (args.config)

    # TODO: Create MLP model with Keras layers. Train on a window of samples, up to some max window size.
    # TODO: Want to predict next sample, and continue to do so until next window ready. (Extreme case is window_size=1, which would be a retrain on each new sample ingested.)
    
    # Read contents of file (if provided), else read from stdin.
    #print ( "loading data... ")
    data = None
    if args.filename is not None:
        with open(args.filename, "r") as f:
            data = json.loads(f.read()) 
    else:
        text = "".join([line for line in sys.stdin])
        data = json.loads(text)

    x = np.array(data["acceleration_data"])
    input_sample_rate = np.float(data["accelerometer_sample_rate"])


    # make sure to reset x and y since we modify them for resampling
    x = np.array(data["acceleration_data"])

    epochs = args.epochs
    units = [int(x) for x in args.units]
    prediction_time = 1
    training_window = args.training_window
    history_length = args.history
    prev_model = None
    use_gpu = args.use_gpu

    y_predicted = [0 for x in range(0, training_window)] # Pack with zeros for first training window.
    y_autopredicted = [0 for x in range(0, training_window)] # Pack with zeros for first training window.

    # Online training fun
    #print("Number of windows: {}".format(len(range(0, len(x)-prediction_time-training_window, training_window))))
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
                y_predicted += y_pred.flatten().tolist()

        # Normalize the training data.
        # TODO

        # Train the model + get predictions.
        model = train_model(units, x_train, y_train, history_length, epochs=epochs, want_gpu=use_gpu)
        #print("x_train shape: {}".format(x_train.shape))
        y_pred = model.predict(x_train)

        prev_model = model # Swap in the last window's model.
        
        # the following line causes a runtime memory error, so I needed to expand the statement into a loop
        #rmse = math.sqrt(np.mean(np.square(y_train_pred - y_train)))
        # hacked version to avoid memory error:
        size = y_pred.shape[0]
        diffs = np.ndarray((size,), float)
        for i in range(0, size):
            diffs[i] = np.square(y_pred[i] - y_train[i])
        rmse = math.sqrt(np.mean(diffs))
        #print("RMSE {}".format(rmse))

        # AUTO-PREDICTION
        # Attempt to predict the entire next window by iteratively predicting forward in time.
        future_x = np.array([x_train[-1]])
        j = 0
        for i in range(0, training_window):
            y_pred = model.predict([future_x])
            #print("Prediction: {}".format(y_pred))
            future_x = np.hstack((future_x[:,1:], y_pred)) # Move the window up 1 sample.
            j += 1
            # Append predicted samples to history list.
            if j == history_length:
                y_autopredicted += future_x.flatten().tolist()
                j = 0

    y_predicted += [0 for x in range(0, training_window)]

    # Computed global RMS.
    size = len(y_predicted)-(training_window*2)
    diffs = np.ndarray((size,), float)
    for i in range(0, size):
        diffs[i] = np.square(y_predicted[i+training_window] - x[i+training_window])
    rmse = math.sqrt(np.mean(diffs))
    print("Global RMSE: {}".format(rmse))

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
        axs[1].plot(x_data, y_data, '-')
        axs[1].set_ylabel('Predicted (w/real data)')

        #y_data = y_autopredicted
        #axs[2].plot(x_data, y_data, '-')
        #axs[2].set_ylabel('Auto-predicted')

        # Put a title on the plot and the window, then render.
        fig.suptitle('(MLP) Original vs Predicted signals', fontsize=15)
        fig.canvas.set_window_title('MLP Predicted Signals')
        plt.show()


if __name__ == '__main__':
    main()
