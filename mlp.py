#!/usr/bin/python3
# MLP/ANN model script.

# Usage:
#   python3 mlp.py CONFIG_FILE [DB_NAME.sqlite]
#
# Description:
#  -- JSON Input Format --
#   {
#     "acceleration_data": [float, float, ...],
#     "accelerometer_sample_rate": int
#   }
#
#  -- Database --
#   If no database name is specified, the script will look for one with
#   the same name as the configuration file, creating a new SQLite3 database
#   if no database by that name already exists.
#
#  -- Accessing the DB --
#   Access to the database goes through the PonyORM library. This makes
#   constructing complex queries very concise in terms of code, and the
#   author (Philip) has some prior professional experience using it.
#   Please, for the love of all that is good in this world,
#   DO NOT USE RAW SQL QUERIES on the database from this script.
#   (External scripts running read-only queries are fine.)
#
#  -- DB Schema --
#   The script maintains 1 table in the database: Results.
#   The table contains result values from online training.
#   The schema is encoded in the `Result` class below.
#   Schemas will be unique to the algorithm type being trained (MLP/LSTM/etc).
#
#  -- Resumption --
#   Rows that have `finished = FALSE` are being attempted by a job process.
#   If a job fails for any reason, there will be leftover rows in the
#   database that have the `finished = FALSE` property.
#   On startup, any rows with this property are run first, one-at-a-time, to
#   try to give each job the best chance at finishing, since they clearly
#   did not the first time around.
#
#  -- Multiprocessing --
#   Multithreading in Python doesn't work very well, due to the GIL.
#   As a result, we plan to use the jankier `multiprocessing` module to get
#   stuff done. The issue of runaway "orphan" processes will likely be solved
#   with a script that kills off all `mlp.py` jobs that are still running.
#   This feature is not finished yet!
#

# --------------------------------------------------------
# Imports & Global configuration

import os
import sys
import math
import json
import collections
import argparse
from datetime import datetime
import configparser
from pony.orm import db_session, Database, commit, count, PrimaryKey, Required, Optional
import numpy as np
import scipy.signal as signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clear_session
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Used to silence tensorflow banner printout.
import tensorflow as tf  # noqa: E402
# TensorFlow dumps too much debug info at startup. Hush it up.
# Cite: https://stackoverflow.com/a/38645250
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)

default_job_dict = {
    'author': 'unknown',
    'dataset': '',
    'downsample-factor': '1.0',
    'layers': '',
    'activations': '',
    'epochs': '',
    'training-window': '',
    'sample-window': '',
    'forecast-length': '1',
    'prediction-gap': '0',
    'runs-per-parameter-set': '1',
    'nprocs': '1',
}


# --------------------------------------------------------
# CLI Utility Functions

def parse_args():
    parser = argparse.ArgumentParser(description='ANN/MLP Model')
    parser.add_argument('config', type=str, help="Configuration filename. (ex: config.ini)")
    parser.add_argument('database', nargs='?', type=str, default='db.sqlite', help="SQLite3 database filename. (ex: db.sqlite)")

    return parser.parse_args()


# --------------------------------------------------------
# Config File Utility Functions

# Read and parse config file.
def read_config_ini(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


# Utility function for parsing range values from a string:
# str -> [int]
def parse_ranges(v):
    # Split by commas, then sub-split along .. ranges.
    values = []
    parts = v.split(',')
    for part in parts:
        # Handle range syntax.
        if '..' in part:
            rv = [int(x) for x in part.split('..')]
            if len(rv) < 3:
                values += [x for x in range(rv[0], rv[1])]
            else:
                values += [x for x in range(rv[0], rv[1], rv[2])]
        else:
            # Otherwise, assume it's a single value.
            values.append(int(part))
    return values


# Override default values for each section, and return a dictionary of
# dictionaries.
# ConfigParser.SafeConfigParser -> dict(str: dict)
def override_defaults(config, default_section_dict):
    out = {}
    required_keys = ["training-window", "sample-window", "epochs", "dataset", "layers", "activations"]
    for section in config.sections():
        out[section] = dict(default_section_dict)
        for (k, v) in config.items(section):
            if k in out[section]:
                out[section][k] = v
            else:
                print("Unrecognized key '{}' in section '{}'".format(k, section), file=sys.stderr)

        # Ensure that required keys are present, since we can't do
        # anything useful without them.
        for k in required_keys:
            assert k in out[section], "Missing required key '{}' in section '{}'".format(k, section)

        # Fixup the types for the different parameters.
        for k in out[section]:
            v = out[section][k]
            #print("{}:{}".format(k,v), file=sys.stderr)  # DEBUG
            if k == 'layers':
                out[section][k] = [int(x) for x in v.split(',')]  # String split + int
            elif k == 'activations':
                out[section][k] = v.split(',')  # Just string split
            elif k == 'downsample-factor':
                out[section][k] = float(v)
            elif k == 'training-window':
                out[section][k] = parse_ranges(v)
            elif k == 'sample-window':
                out[section][k] = parse_ranges(v)
            elif k == 'epochs':
                out[section][k] = parse_ranges(v)
            elif k == 'forecast-length':
                out[section][k] = parse_ranges(v)
            elif k == 'prediction-gap':
                out[section][k] = parse_ranges(v)
            elif k == 'runs-per-parameter-set':
                out[section][k] = int(v)
            elif k == 'nprocs':
                out[section][k] = int(v)
    return out


# --------------------------------------------------------
# DB Utility Functions

db = Database()


class Result(db.Entity):
    id = PrimaryKey(int, auto=True)
    author = Required(str)
    dataset = Required(str)
    downsample_factor = Required(float, default=1.0)
    layers = Required(str)
    activations = Required(str)
    epochs = Required(int)
    training_window = Required(int)
    sample_window = Required(int)
    forecast_length = Required(int)
    prediction_gap = Required(int)
    creation_ts = Required(datetime, default=lambda: datetime.utcnow())
    rmse_global = Optional(float)
    finished = Required(bool, default=False)

    # TODO: Fill in the other fields.
    def to_json(self):
        out = {
            "creation_ts": self.creation_ts.isoformat() + 'Z',
        }
        return out


# [Any] -> str
def cannonicalize(ls):
    return ",".join([str(x) for x in ls])


# --------------------------------------------------------
# ML Functions

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

        # HACK: This is to prevent null windows from happening, which is apparently a flavor of janky indexing we can have happen here.
        if len(x_train) > 0 and len(y_train) > 0:
            out.append((i, x_train, y_train))

    return out


def create_model(layers, activations, sampling_window, output_layer_size=1, want_verbose=0):
    model = Sequential()
    top_layer = True
    layers = list(layers)
    activations = list(activations)  # Clone the list so it can be consumed.
    # If provided a list, it will create 1 layer per integer list item.
    if isinstance(layers, collections.abc.Iterable):
        for u in layers:
            if top_layer:
                model.add(Dense(u, input_shape=(sampling_window,), activation=activations.pop(0)))
                top_layer = False
            else:
                model.add(Dense(u, activation=activations.pop(0)))
    # Otherwise, it will just make a single hidden layer.
    else:
        model.add(Dense(layers, input_shape=(sampling_window,), activation=activations[0]))
    model.add(Dense(output_layer_size))  # Default: 1
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def train_model(layers,
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
                model = create_model(layers, activations, sampling_window, forecast_length)
                model.fit(x_train, y_train, epochs=epochs, verbose=want_verbose)
                # Reset states for LSTMs here.
        else:
            model = create_model(layers, activations, sampling_window, forecast_length)
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


# This used to be the guts of the `main` function; it's now its own function.
# Returns a dictionary of results.
def run_model(raw_dataset, layers, activations, epochs, training_window, sample_window,
              forecast_length, prediction_gap,
              use_gpu=False, want_verbose=0, online=False):
    x = np.array(raw_dataset, copy=True)
    # Ensure every hidden layer gets an explicit activation function.
    if len(layers) > len(activations):
        len_diff = len(layers) - len(activations)
        activations += ["linear" for x in range(0, len_diff)]
    model = None
    prev_model = None

    y_predicted = np.zeros((len(x),))
    y_error = [0] * training_window
    results_idx = training_window

    # Online training fun
    training_examples = generate_training_samples(x, training_window, sample_window, prediction_gap, forecast_length)  # noqa E501
    #print("train_examples: {}".format([(i, len(x), len(y)) for i,x,y in training_examples]))  # DEBUG

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
        if online and model is not None:
            model = train_model(layers, activations, x_train, y_train, sample_window, forecast_length, epochs=epochs, want_gpu=use_gpu, model=model, want_verbose=want_verbose)
        else:
            model = train_model(layers, activations, x_train, y_train, sample_window, forecast_length, epochs=epochs, want_gpu=use_gpu, want_verbose=want_verbose)
        y_pred = model.predict(x_train)

        prev_model = model  # Swap in the last window's model.

        # Clean up all excess state Keras is keeping around.
        clear_session()

    # Computed global RMSE.
    size = len(y_predicted)-(training_window*2)
    diffs = np.ndarray((size,), float)
    for i in range(0, size):
        diffs[i] = np.square(y_predicted[i+training_window] - x[i+training_window])
    rmse = math.sqrt(np.mean(diffs))
    y_error += diffs.tolist()
    # print("Global RMSE: {}".format(rmse))
    y_error += [0] * training_window

    # Return dictionary for other functions to consume.
    out = {
        "activations": ",".join([str(x) for x in activations]),
        "creation_ts": datetime.utcnow().isoformat(),
        "forecast_length": forecast_length,
        "prediction_gap": prediction_gap,
        "training_window": training_window,
        "sample_window": sample_window,
        "epochs": epochs,
        "layers": ",".join([str(x) for x in layers]),
        "rmse_global": rmse,
        "metadata": json.dumps({}),
    }
    return out


# --------------------------------------------------------
# Grid Search Function

# Takes a single bundle of search parameters, and runs every permutation
# of parameters the required number of times, saving the results into a
# database.
# Upon "resuming" a search (such as in the case of a killed job), this
# function will keep skipping forward until it hits whatever part of the
# search was left undone.
# This monster wound up much longer than I'd like, due to the looong queries.
def grid_search(config_dict, raw_dataset):
    author = config_dict['author']
    dataset = config_dict['dataset']
    downsample_factor = config_dict['downsample-factor']
    layers = config_dict['layers']
    activations = config_dict['activations']
    epochs_list = config_dict['epochs']
    training_window_list = config_dict['training-window']
    sample_window_list = config_dict['sample-window']
    forecast_length_list = config_dict['forecast-length']
    prediction_gap_list = config_dict['prediction-gap']
    num_runs_per_parameter_set = config_dict['runs-per-parameter-set']
    # Print diagnostic info so user can bail if they made a mistake.
    total_required_runs = (len(epochs_list) * len(training_window_list) *
                           len(sample_window_list) * len(forecast_length_list) *
                           len(prediction_gap_list) * num_runs_per_parameter_set)
    print("This parameter set will require up to {} runs.".format(total_required_runs), file=sys.stderr)
    iter_counter = 0
    run_counter = 0
    # Actual search loop.
    for epochs in epochs_list:
        for training_window in training_window_list:
            for sample_window in sample_window_list:
                if sample_window > training_window:
                    continue
                for forecast_length in forecast_length_list:
                    for prediction_gap in prediction_gap_list:
                        # Implicit else:
                        # Query database to see if we have data from enough runs.
                        # Note: If the `with` statement is too slow, we can
                        #   use the `@db_session` decorator on a function
                        #   instead for more precise release of DB sessions.
                        row_count = 0
                        with db_session:
                            row_count = count(r for r in Result
                                              if r.dataset == dataset
                                              and r.downsample_factor == downsample_factor
                                              and r.layers == cannonicalize(layers)
                                              and r.activations == cannonicalize(activations)
                                              and r.epochs == epochs
                                              and r.training_window == training_window
                                              and r.sample_window == sample_window
                                              and r.forecast_length == forecast_length
                                              and r.prediction_gap == prediction_gap
                                              and r.finished)
                        # Not enough runs? We can fix that.
                        if row_count < num_runs_per_parameter_set:
                            with db_session:
                                # Insert an unfinished row for now. We'll update it later.
                                record = Result(author=author,
                                                dataset=dataset,
                                                downsample_factor=downsample_factor,
                                                layers=cannonicalize(layers),
                                                activations=cannonicalize(activations),
                                                epochs=epochs,
                                                training_window=training_window,
                                                sample_window=sample_window,
                                                forecast_length=forecast_length,
                                                prediction_gap=prediction_gap,
                                                finished=False)
                                commit()  # Ensure we flush the record to disk.
                                # Run the model with the provided parameters.
                                result = run_model(raw_dataset, layers, activations, epochs, training_window, sample_window, forecast_length, prediction_gap, use_gpu=False, want_verbose=False, online=False)
                                # Now we update the row.
                                record.finished = True
                                record.rmse_global = result["rmse_global"]
                                commit()
                                # Provide occasional status messages so that the user doesn't think the job is hung up.
                                run_counter += 1
                                if run_counter % 1000 == 0:
                                    print("\nCompleted {} runs so far. At {}/{}".format(run_counter, iter_counter, total_required_runs), file=sys.stderr)
                        iter_counter += 1
                        print('.', end='', file=sys.stderr)
                        sys.stderr.flush()

    print('', file=sys.stderr)  # Print a trailing newline for formatting.


# --------------------------------------------------------
# Main Function

def main():
    args = parse_args()
    print("Loading config file: '"+args.config+"'", file=sys.stderr)
    config = read_config_ini(args.config)
    parameter_sets = override_defaults(config, default_job_dict)

    # Connect to DB and auto-gen tables as needed.
    db.bind(provider='sqlite',
            filename=args.database,
            create_db=True)
    db.generate_mapping(create_tables=True)
    print("Connected to database: '{}'".format(args.database), file=sys.stderr)

    # Perform the horrifying grid search here.
    for k in parameter_sets:
        params = parameter_sets[k]
        filename = params['dataset']
        downsample_factor = params['downsample-factor']
        # Load up dataset before each search:
        data = None
        with open(filename, "r") as f:
            data = json.loads(f.read())
        x = np.array(data["acceleration_data"])
        # Downsampling occurs here, before we get into the grid search.
        if downsample_factor != 1.0:
            x = np.array(signal.resample(x, int(x.shape[0] / downsample_factor)))
        #input_sample_rate = np.float(data["accelerometer_sample_rate"])
        print("Starting run for: '{}'".format(k), file=sys.stderr)
        grid_search(params, x)

    # Plot only if the user asked for plotting.
    #if args.plot:
    #    import matplotlib.pyplot as plt  # Intentionally late import.

    #    #num_plots = 2
    #    num_plots = 1
    #    x_data = np.arange(len(x))

    #    # Plot subplot for each signal.
    #    fig, axs = plt.subplots(num_plots, 1, sharex=True)
    #    ax2 = axs.twinx()
    #    y_data = x
    #    axs.plot(x_data, x, 'r-', label="Signal")
    #    axs.plot(x_data, y_predicted, 'b-', label="Predicted")
    #    ax2.plot(x_data, y_error, 'g-', label="Error")
    #    axs.set_ylabel('Signal')
    #    ax2.set_ylabel("Error")

    #    #y_data = y_predicted
    #    #axs[1].plot(x_data, y_data, 'r-')
    #    #axs[1].plot(x_data, x, 'b-')
    #    #axs[1].set_ylabel('Predicted (w/real data)')
    #    fig.legend() # Captures labels from entire plot.

    #    # Put a title on the plot and the window, then render.
    #    #fig.suptitle('(MLP) Original vs Predicted signals', fontsize=15)
    #    #fig.canvas.set_window_title('MLP Predicted Signals')
    #    plt.show()


if __name__ == '__main__':
    main()
