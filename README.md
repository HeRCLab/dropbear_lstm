# LSTM model of the DROPBEAR experiment

An attempt to develop a Python version of the previous Matlab model.

Trains many times slower than the equivalent Matlab model.  I'm not sure why.
In fact it is currently so slow that I have not yet been able to debug the plotting code.
Need to make Tensorflow use the GPU.

## Usage:
* Modify sweep parameters as needed near top of file.

* Run with:
`./dropbear_lstm.py

## Requirements
* TensorFlow
* Keras
