#!/usr/bin/python3

#import IPython as IP

import numpy as np
import json
import math
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
import matplotlib.pyplot as plt
import argparse
import yaml
import tensorflow as tf     # issue with importing tensorflow on
                            # tachyon due to its lack of AVX instructions
                            # NOTE:  must use on higgs for GPU

def parse_args():
    parser = argparse.ArgumentParser(description='DROPBEAR model')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='config.yaml', help='configuration file')
    return parser.parse_args()

def create_model (units,batchsize):
    model = Sequential()
    model.add(LSTM(units,input_shape=(1,1),batch_size=batchsize,stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model

def load_config (configfile):
    with open(configfile) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    ### SWEEP PARAMETERS:  modify as needed
    # TODO: check for missing values and assign defaults

    # how much to downsample the acceleration data
    downsample_low = config['downsample']['start']
    downsample_high = config['downsample']['end']
    downsample_increment = config['downsample']['increment']
    downsample_levels = np.arange(downsample_low,downsample_high+downsample_increment,downsample_increment)

    # how much history to keep in the lstm in terms of units of 25 samples at
    # the current acceleration sample rate.  thus,
    # # units = 25 * history
    # lstm memory in seconds = 25 * history / sample rate
    history_low = config['history']['start']
    history_high = config['history']['end']
    history_increment = config['history']['increment']
    history_lengths = np.arange(history_low,history_high+history_increment,history_increment)
    
    # number of samples to time shift the acceleration relative to the pin
    # location.  in other words, how much time into the future are we predicting
    # the pin location given the acceleration (NOTE: see below, the implementation
    # of this might be bugged)
    time_shift = config['time_shift']

    # how much of the training data should be visible to the training algorithm,
    # i.e. what is the training/testing data split.
    training_portion = np.float(config['training_portion'])
    return downsample_levels, history_lengths, time_shift, training_portion

def train_model(units,x_train,y_train):
    # the following line verifies that Tensorflow has detected a target GPU
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    batchsize = x_train.shape[0]
    #batchsize = 1

    with tf.device('/gpu:0'): 
        model = create_model(units,batchsize)
        model.fit(x_train,y_train,batch_size=batchsize,epochs=10,verbose=1)
        model.reset_states()

    y_train_pred_norm = model.predict(x_train,batch_size=batchsize)

    return y_train_pred_norm

def main():
    #IP.get_ipython().magic('reset -sf')
    args = parse_args()
    downsample_levels, history_lengths, time_shift, training_portion = load_config (args.config)

    # size of the space for parameter sweep
    arrsz = [len(downsample_levels), len(history_lengths)]

    # allocate storage to track storage, throughput, and accuracy for trained nets
    storage    = np.zeros(arrsz)
    throughput = np.zeros(arrsz)
    rmse_res   = np.zeros(arrsz)
    
    # declare data outside of with-block to avoid losing it outside of the with scope
    data = None
    
    # open the data file and load the contents into data
    print ( "loading data... ")
    with open("data_6_with_FFT.json", "r") as f:
        data = json.loads(f.read()) 
    
    # print informational messages about the sample rates of the pin location
    # and the vibration
    input_sample_rate = np.float(data["accelerometer_sample_rate"])
    print("input_sample_rate: {}".format(input_sample_rate))
    output_sample_rate = len(data["measured_pin_location"]) / data["measured_pin_location_tt"][-1]
    print("output_sample_rate: {}".format(output_sample_rate))
    
    # create a mesh grid for plotting
    [xg, yg] = np.meshgrid(
        input_sample_rate / downsample_levels,
        history_lengths * 250 / input_sample_rate
    );
    
    print("meshgrid shapes: {}, {}".format(xg.shape, yg.shape))
    
    # used to log values in mesh grid
    dd=0
    hh=0
    fignum=1

    sample_rates = input_sample_rate / downsample_levels
    
    print("sample rates = ")
    print(sample_rates)

    num_units = np.zeros([len(sample_rates),len(history_lengths)])
    for i in range(0,len(sample_rates)):
        for j in range(0,len(history_lengths)):
            num_units[i,j] =  25 / input_sample_rate * sample_rates[i] * history_lengths[j]

    print("number of LSTM units for parameter search =")
    print(num_units)

    # make sure to reset x and y since we modify them for resampling
    x = np.array(data["acceleration_data"])
    y = np.array(data["measured_pin_location"])

    # remove nans from acceleration data
    # TODO: might want to interpolate / make more robust
    for i in range(y.shape[0]):
        if math.isnan(y[i]):
            y[i] = y[i-1]

    for downsample in downsample_levels:
        for history in history_lengths:
    
            # downsample acceleration data
            #x = signal.decimate(x,downsample)
            x_resamp = signal.resample(x,int(x.shape[0]/downsample))
    
            # upsample pin location data
            #frac = fr.Fraction(x.shape[0]/y.shape[0]).limit_denominator(1000)
            #y = signal.resample(y,frac.numerator)
            #y = signal.decimate(y,frac.denominator)
            y_resamp = signal.resample(y,x_resamp.shape[0])
    
            # NOTE: this might not work as intended, since shifting the acceleration
            # data to the right effectively shifts the pin location data to the left,
            # making it so we predict the past, not the future as originally
            # was the intent here 
            x_resamp = x_resamp[time_shift:]
            y_resamp = y_resamp[0:y_resamp.shape[0]-time_shift]
    
            # extract the training set from x and y
            training_samples = int(training_portion*x_resamp.shape[0])
            x_train = x_resamp[0:training_samples]
            y_train = y_resamp[0:training_samples]
    
            if training_portion == 1:
                x_test = [];
                y_test = [];
            else:
                x_test = x_resamp[training_samples:]
                x_test = y_resamp[training_samples:]
            
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
            x_test_norm = (x_test - mu_x) / sig_x
            y_test_norm = (y_test - mu_y) / sig_y

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
            history_length = 250 / input_sample_rate * history
            numHiddenUnits = int(history_length*current_sample_rate)
            
            # train the model
            y_train_pred_norm = train_model(numHiddenUnits,x_train_norm,y_train_norm) 
 
            # un-normalize predicted data
            y_train_pred = y_train_pred_norm * sig_y + mu_y;

            # the following line causes a runtime memory error, so I needed to expand the statement into a loop
            #rmse = math.sqrt(np.mean(np.square(y_train_pred - y_train)))
            # hacked version to avoid memory error:
            size = y_train_pred.shape[0]
            diffs = np.ndarray((size,),float)
            for i in range(0,size):
                diffs[i]=np.square(y_train_pred[i] - y_train[i])
            rmse = math.sqrt(np.mean(diffs))
            
            rmse_res[dd,hh]=rmse;
            
            f = plt.figure(fignum)
            fignum = fignum+1
            plt.subplot(3,1,1)
            plt.plot(y_train)
            plt.title('y train')
            plt.xlabel('time')
            plt.ylabel('position')

            plt.subplot(3,1,2)
            plt.plot(y_train_pred)
            plt.xlabel('time')
            plt.ylabel('position')
            plt.title("prediction for sample rate = %0.2f Hz and %d units" % ((input_sample_rate/downsample),numHiddenUnits))

            plt.subplot(3,1,3)
            # another memory error...
            #plt.plot(y_train_pred - y_train)
            for i in range(0,size):
                diffs[i]=y_train_pred[i] - y_train[i]
            plt.plot(diffs)
            plt.xlabel('time')
            plt.ylabel('error')
            plt.title("RMSE = %0.2f" % rmse)
            f.show()

            hh=hh+1
        dd=dd+1
        hh=0

    plt.surf(xg,yg,rmse_res)
    plt.colorbar();
    plt.show() 

if __name__ == '__main__':
    main()

