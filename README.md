# LSTM model of the DROPBEAR experiment

An attempt to develop a Python version of the previous Matlab model.

Trains many times slower than the equivalent Matlab model.  I'm not sure why.
In fact it is currently so slow that I have not yet been able to debug the plotting code.
Need to make Tensorflow use the GPU.

## Usage:

 * Modify sweep parameters as needed near top of file.

 * Run with:

    ./dropbear_lstm.py

## Requirements

 * TensorFlow
 * Keras

------

# Wave generator script

This script generates uniformly-sampled synthetic acceleration data.
It can combine signals of multiple frequencies together, and can add noise to a signal as well.

## Usage

Example:

    python3 wavegen.py -n normal -nm 0.3 -p 16000 16000 1 3 5 10 15 > 1_3_5_10_15.json

This will generate a signal with:

 - Number of samples: 16,000
 - Sampling rate: 16 kHz
 - Noise type: normal distribution
 - Noise magnitude: 30% of signal
 - Plots desired (the `-p` option)
 - Frequencies: 1, 3, 5, 10, 15 Hz

Note: Currently signals cannot have different amplitudes or phases.
They are all assumed to start at the same origin, and have the same phase offset.
This behavior may be adjusted in a future version of the script.

-----

# Keras online multi-layer perceptron

This script is a Keras reimplementation of the `online_lstm.m` script, specifically the MLP parts of the script.
Where possible, indexing and wrangling of data was kept identical to the MATLAB source material, although some adjustments were required to account for Numpy differences, and for Python starting indexing at 0 instead of 1.

It currently has some (limited) plotting capabilities, which may be expanded later.

## Usage

      python3 mlp.py -u 20 -u 10 --history-length 10 -w 40 -e 100 -p -f example-data.json

This will result in:

 - MLP Layers: *input* -> 20 neurons -> 10 neurons -> *1 neuron* (italics for implictly-added layers)
 - History length: 10 samples
 - Training window: 40 samples
 - Epochs for training: 100
 - Plots desired (the `-p` option)
 - File to load from: `example-data.json`

Sampling rate and training are handled automatically, based on the input data.

Input data can also be provided over `stdin`, allowing this script to be chained with the `wavegen.py` signal generator.
