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
    parser.add_argument('-u', '--units', action="append", default=[1], help="Number of units in hidden layer. Providing this argument more than once will add additional hidden layers. (default: 1)")
    parser.add_argument('--history', '--history-length', type=int, default=10, help="Number of samples in the history buffer. (default: 10)")
    parser.add_argument('-w', '--training-window', type=int, default=40, help="Number of samples in each training window. (default: 40)")
    parser.add_argument('-f', '--filename', type=str, help="Filename of JSON waveform data to read in.")
    parser.add_argument('-t', '--training-ratio', type=float, help="Ratio of incoming data to use for training, on a scale of 0.0-1.0 (default: 0.1)")

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
    model.compile(loss='mse',optimizer='adam')
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
            model.fit(x_train, y_train, epochs=epochs, verbose=1)
            #model.reset_states() # Only needed for stateful LSTM.
    else:
        model = create_model(units, history_length)
        model.fit(x_train, y_train, epochs=epochs, verbose=0)

    y_train_pred_norm = model.predict(x_train)
    return y_train_pred_norm


def main():
    args = parse_args()
    #downsample_levels, history_lengths, time_shift, training_portion = load_config (args.config)

    # TODO: Create MLP model with Keras layers. Train on a window of samples, up to some max window size.
    # TODO: Want to predict next sample, and continue to do so until next window ready. (Extreme case is window_size=1, which would be a retrain on each new sample ingested.)
    
    # Read contents of file (if provided), else read from stdin.
    print ( "loading data... ")
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

    units = [int(x) for x in args.units]
    prediction_time = 1
    training_window = args.training_window
    history_length = args.history

    # Online training fun
    print("Number of windows: {}".format(len(range(0, len(x)-prediction_time-training_window, training_window))))
    for i in range(0, len(x)-prediction_time-training_window, training_window):
        start_train = i
        end_train = i + training_window
        x_train = []
        y_train = []

        for j in range(0, training_window):
            start_idx = i - history_length + j
            if start_idx < 1:
                continue
            end_idx = i + j
            x_train += [x[start_idx:end_idx]]
            y_train += [x[end_idx+prediction_time]]
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Normalize the training data.
        # TODO

        # Train the model + get predictions.
        y_pred = train_model(units, x_train, y_train, history_length)
        
        # the following line causes a runtime memory error, so I needed to expand the statement into a loop
        #rmse = math.sqrt(np.mean(np.square(y_train_pred - y_train)))
        # hacked version to avoid memory error:
        size = y_pred.shape[0]
        diffs = np.ndarray((size,), float)
        for i in range(0, size):
            diffs[i] = np.square(y_pred[i] - y_train[i])
        rmse = math.sqrt(np.mean(diffs))
        print("RMSE {}".format(rmse))


if __name__ == '__main__':
    main()
