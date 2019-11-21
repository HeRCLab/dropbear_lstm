#!/usr/bin/python3

import numpy as np
import json
import math
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
import matplotlib.pyplot as plt
import tensorflow as tf    # issue with importing tendoeflow on
                            # tachyon due to its lack of AVX instructions

### SWEEP PARAMETERS:  modify as needed

# how much to downsample the acceleration data
downsample_levels = np.arange(8,60,8)

# how much history to keep in the lstm in terms of units of 25 samples at
# the current acceleration sample rate.  thus,
# # units = 25 * history
# lstm memory in seconds = 25 * history / sample rate
history_lengths = np.arange(1,10,1)

# number of samples to time shift the acceleration relative to the pin
# location.  in other words, how much time into the future are we predicting
# the pin location given the acceleration (NOTE: see below, the implementation
# of this might be bugged)
time_shift = 0

# how much of the training data should be visible to the training algorithm,
# i.e. what is the training/testing data split.
training_portion = np.float(1);
 
def create_model (units):
    model = Sequential()
    model.add(LSTM(units,input_shape=(1,1),batch_size=1,stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model

def main():   
    # size of the space for parameter sweep
    arrsz = [len(downsample_levels), len(history_lengths)]

    # allocate storage to track storage, throughput, and accuracy for trained nets
    storage    = np.zeros(arrsz)
    throughput = np.zeros(arrsz)
    rmse_res   = np.zeros(arrsz)
    
    # declare data outside of with-block to avoid losing it outside of the with scope
    data = None
    
    # open the data file and load the contents into data
    with open("data_6_with_FFT.json", "r") as f:
        data = json.loads(f.read())
    
    print ( "loading data... ")
    
    # print informational messages about the sample rates of the pin location
    # and the vibration
    input_sample_rate = np.float(data["accelerometer_sample_rate"])
    print("input_sample_rate: {}".format(input_sample_rate))
    output_sample_rate = len(data["measured_pin_location"]) / data["measured_pin_location_tt"][-1]
    print("output_sample_rate: {}".format(output_sample_rate))
    
    # create a mesh grid for plotting (why here?)
    [xg, yg] = np.meshgrid(
        input_sample_rate / downsample_levels,
        history_lengths * 25 / input_sample_rate
    );
    
    print("meshgrid shapes: {}, {}".format(xg.shape, yg.shape))
    
    for downsample in downsample_levels:
        for history in history_lengths:
            # make sure to reset x and y since we modify them for resampling
            x = np.array(data["acceleration_data"])
            y = np.array(data["measured_pin_location"])
    
            # remove nans from acceleration data
            # TODO: might want to interpolate / make more robust
            for i in range(y.shape[0]):
                if math.isnan(y[i]):
                    y[i] = y[i-1]
    
            # downsample acceleration data
            #x = signal.decimate(x,downsample)
            x = signal.resample(x,math.floor(x.shape[0]/downsample))
    
            # upsample pin location data
            #frac = fr.Fraction(x.shape[0]/y.shape[0]).limit_denominator(1000)
            #y = signal.resample(y,frac.numerator)
            #y = signal.decimate(y,frac.denominator)
            y = signal.resample(y,x.shape[0])
    
            # NOTE: this might not work as intended, since shifting the acceleration
            # data to the right effectively shifts the pin location data to the left,
            # making it so we predict the past, not the future as originally
            # was the intent here 
            x = x[time_shift:]
            y = y[0:y.shape[0]-time_shift]
    
            # make the x and y array have the same length, needed because of the
            # differing simulation times for both
            smallest = min(y.shape[0],x.shape[0])
            x = x[0:smallest]
            y = y[0:smallest]
    
            # extract the training set from x and y
            training_samples = math.floor(training_portion*x.shape[0])
            x_train = x[0:training_samples]
            y_train = y[0:training_samples]
    
            if training_portion == 1:
                x_test = [];
                y_test = [];
            else:
                x_test = x[training_samples:]
                x_test = y[training_samples:]
            
            # get mu and signal for x and y
            # note:  x and y are type sp.ndarray after having been returned
            # from sp.decimate().  do we need to typecast them to np.array?
            # answer:  apparently not, since according to the internets, these
            # types are the same, since array() only actually creates an
            # initializes ndarray.
            # is this array type compatible with tensorflow training?  this remains
            # to be seen...
            mu_x = x_train.mean()
            mu_y = y_train.mean()
            sig_x = x_train.std()
            sig_y = y_train.std()
    
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)
            
            # normalize
            x_train_norm = (x_train - mu_x) / sig_x
            y_train_norm = (y_train - mu_y) / sig_y

            # need to reshape data due to some inconvenient and seemingly nonsensical
            # requirement of keras's lstm layer
            reshape_3 = lambda x: x.reshape((x.shape[0], 1, 1))
            x_train_norm = reshape_3(x_train_norm)
            x_test_norm = reshape_3(x_test_norm)
        
            reshape_2 = lambda x: x.reshape((x.shape[0], 1))
            y_train_norm = reshape_2(y_train_norm)
            y_test_norm = reshape_2(y_test_norm)
 
            # converted parameters
            current_sample_rate = input_sample_rate / downsample
            history_length = 25 / input_sample_rate * history
            numHiddenUnits = math.floor(history_length*current_sample_rate)
    
            model = create_model(numHiddenUnits)
            model.fit(x_train_norm,y_train_norm,batch_size=1,epochs=10,verbose=1)
            model.reset_states()
    
            y_train_pred_norm = model.predict(x_train_norm,batch_size=1)
            
            # un-normalize predicted data
            
            y_train_pred = y_train_pred_norm * sig_y + mu_y;
            rmse = math.sqrt(np.mean(np.square(y_train_pred - y_train)))

            plt.subplot(3,1,1)
            plt.plot(y_train)
            plt.title('y train')
            plt.xlabel('time')
            plt.ylabel('position')
            plt.subplot(3,1,2)
            plt.plot(ypred)
            plt.ylabel('position')
            plt.title('prediction')
            plt.subplot(3,1,3)
            plt.plot(ypred - y_train)
            plt.xlabel('time')
            plt.ylabel('error')
            plt.title('RMSE = ' + rmse)
            plt.show()

if __name__ == '__main__':
        main()
        
