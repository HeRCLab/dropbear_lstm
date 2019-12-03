# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:40:37 2019

@author: barzegar
"""
def generateData (frequency = .1, duration=1000, dt=.001, seq_len=50,delay=2,
                  n_training_data=500, n_test_data=100, noise=0):
    '''
    generates sin(wt) for a duration of duration and with time step of dt
    frequency is the frequency if the sine wave
    seq_len is the length of the training data
    delay is the sampling frequency
    '''
    #create the sine wave
    t = np.arange(0, duration, dt)
    random_phase = np.random.uniform(-np.pi*2,+np.pi*2)
    noise = np.array([np.random.uniform(-1,1)*.6 for x in t])
    data = np.sin(2*np.pi*frequency*t+random_phase)
#    print(data.shape)
    
    #sample from the data to get training dataset
    X = np.empty((seq_len, n_training_data))
    Y = np.empty((n_training_data, 1))
    for i in range(n_training_data):
        rand = np.random.randint(0, len(t)-seq_len*delay)
        X[:,i] = data[rand:rand+seq_len*delay:delay]
        Y[i] = data[rand+seq_len*delay]
        
    X_test = np.empty((seq_len, n_test_data))
    Y_test = np.empty((n_test_data, 1))
    for i in range(n_test_data):
        rand = np.random.randint(0, len(t)-seq_len*delay)
        X_test[:,i] = data[rand:rand+seq_len*delay:delay]
        Y_test[i] = data[rand+seq_len*delay]
        
    return X, Y, X_test, Y_test


def animate (i, x):
    ax1.clear()
    ax1.plot(x)
    
def predict(x, y, hidden_dim, seq_len, Why):
    lstm = LSTM()
    for i in range(x.shape[0]*2):
        prev_h = np.zeros((hidden_dim, 1))
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(seq_len):
            xt = np.zeros((x.shape[0]+1,1))
            xt[t] = x[t]
            xt[seq_len] = 1
            I_t = np.concatenate((xt,prev_h), axis=0)
            z_t, s_t, h_t = lstm.doForward(W, I_t, prev_s)
            prev_h = h_t
            prev_s = s_t
        pred = Why.dot(h_t)
        x = np.append(x,pred)
        x = np.delete(x,0)
        y = np.append(y,pred)
    return y
    
    
#ani = animation.FuncAnimation(fig, animate, data=x, interval=1000)
'''
main code
'''    
import timeit
import numpy as np
from RNN_classes import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import ImageMagickWriter
from matplotlib import style
from PIL import Image, ImageDraw

plt.ion()
fig = plt.figure(num=10, figsize=(13.6,4))
ax1 = fig.add_subplot(1,1,1)
#plt.tight_layout()

#style.use('default')

start = timeit.default_timer()
frequency = .1
duration = 500
dt =.001
seq_len = 10
delay = 1000
n_training_data=50
n_test_data=50
noise = 0.0

X, Y, X_test, Y_test = generateData(frequency = frequency, duration = duration,
                                    dt = dt,
                                    seq_len = seq_len, delay=delay,
                                    n_training_data=n_training_data,
                                    n_test_data = n_test_data,
                                    noise = noise)
#X = np.transpose(X)
#plt.plot(X[:,1])
# =============================================================================
#  create RNN
# =============================================================================
learning_rate =0.001
nepoch = 100
# values of the gradients greater thant these will be clipped
max_clip =100
min_clip = -100
# dimension for the two hidden units in out hidden layer, s, and h
hidden_dim = 5
output_dim = 1

Wxh = np.random.normal(0, 1, (hidden_dim, seq_len+1))
Whh = np.random.normal(0, 1, (hidden_dim, hidden_dim))
Why = np.random.normal(0, 1, (output_dim, hidden_dim))

W1 = np.concatenate((Wxh, Wxh, Wxh, Wxh),axis=0)
W2 = np.concatenate((Whh, Whh, Whh, Whh), axis=0)
W = np.concatenate((W1, W2), axis=1)
# calculate loss on training data
prev_loss = 100000;
training_pred=[]

phase = np.pi*.2

t_plot = np.arange(0,650,dt)
data_plot = np.sin(2*np.pi*frequency*t_plot+phase)
x_plot = data_plot[0:seq_len*delay:delay]
x_plot = np.transpose(np.array([x_plot]))
y_plot = data_plot[0:seq_len*delay:delay]
t_scale = t_plot[0:seq_len*delay*3:delay]
images =[]

for epoch in range(nepoch):
    
    if epoch%5==0:
        learning_rate=.005
#        X, Y, _ ,_ = generateData(frequency = frequency, duration = duration,
#                                        dt = dt,
#                                        seq_len = seq_len, delay=delay,
#                                        n_training_data=n_training_data,
#                                        n_test_data = n_test_data,
#                                        noise = noise)
    loss= 0.0
    error = 0.0
    error_i = 0.0
    
    for i in range(Y.shape[0]):
        x, y = X[:,i], Y[i]
        remember_hidden = []
        remember_z = []
        prev_h = np.zeros((hidden_dim, 1))
        prev_s = np.zeros((hidden_dim, 1))
        
                        
        #do a forward pass
        lstm = LSTM()
        for t in range(seq_len):
            xt = np.zeros((x.shape[0]+1,1))
            xt[t] = x[t]
            xt[seq_len] = 1
            I_t = np.concatenate((xt,prev_h), axis=0)
            z_t, s_t, h_t = lstm.doForward(W, I_t, prev_s)
            remember_hidden.append({'h':h_t, 'prev_h':prev_h, 's':s_t, 'prev_s':prev_s})
            remember_z.append({'zt':z_t})
            prev_h = h_t
            prev_s = s_t
        pred = Why.dot(h_t)
        training_pred.append(pred)
        loss_i = (y-pred)**2/2
        loss += loss_i
        error_i = np.abs((pred-y)/y)
        error += error_i
        dW = np.zeros(W.shape)
        # do a backward pass
        dprev_s = np.zeros((hidden_dim, 1))
        dJ_pred = (pred-y)
        dWhy = dJ_pred.dot(remember_hidden[t]['h'].T)
        prev_dWhy = np.zeros(Why.shape)
        prev_dW_i = np.zeros(W.shape)
        G_W = np.zeros((W.shape[0],W.shape[0]))
        for t in reversed(range(seq_len)):
            
            dh = Why.T.dot(dJ_pred)
            s_t = remember_hidden[t]['s']
            prev_s = remember_hidden[t]['prev_s']
            g_t = remember_z[t]['zt'][0]
            i_t = remember_z[t]['zt'][1]
            f_t = remember_z[t]['zt'][2]
            o_t = remember_z[t]['zt'][3]
            forward = [s_t, prev_s, g_t, i_t, f_t, o_t]
            out1, dW_i = lstm.doBackwards(dh, dprev_s, forward, I_t)
            dprev_s = out1[:]
            dW += dW_i
            prev_dW_i = dW_i
            
            #clip large gradients
        if dW.max() > max_clip:
            dW[dW>max_clip] = max_clip
        if dW.min() < min_clip:
            dW[dW<min_clip] = min_clip
#        print(dW)    
        
        Why -= learning_rate*dWhy 
        prev_dWhy = dWhy
        W -= learning_rate*dW
        
    _plot = predict(x_plot, y_plot, hidden_dim, seq_len, Why)
#        ani = animation.FuncAnimation(fig, animate, interval=1000,
#                                      fargs= (_plot,))
    ax1.clear()
    ax1.plot(t_scale,_plot,linestyle='--')
    ax1.plot(t_plot[0:30000],data_plot[0:30000])
    plt.xlabel('time (s)',fontsize=23, fontname="Times New Roman")
    plt.ylabel('value',fontsize=23, fontname="Times New Roman")
    plt.legend(['Prediction', 'True'], loc='upper right',prop={'family':"Times New Roman",'size':20})
    ax1.set_xlim(( 0, 32))
    ax1.set_ylim((-2, 2))
    for tick in ax1.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(20)
        tick.set_fontname('Times New Roman')
    for tick in ax1.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(20)
        tick.set_fontname('Times New Roman')   
        
    plt.pause(.0005)
    ax1.set_facecolor((1,1,1))
    plt.show()
    plt.savefig('fig'+ str(epoch)+'.jpg')
    im = Image.open('fig'+ str(epoch)+'.jpg')
    images.append(im)
    
    
    loss = loss/float(y.shape[0])
    if prev_loss<loss and epoch%5!=0:
        learning_rate = max(learning_rate/2, 1e-7)
        print('lr = ', learning_rate)
    prev_loss = loss
    error = error/float(Y.shape[0])
    
    #calculate loss on the test data
    loss_test = 0.0
    for i in range(Y_test.shape[0]):
        x, y = X_test[:,i], Y_test[i]
        prev_h = np.zeros((hidden_dim, 1))
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(seq_len):
            xt = np.zeros((x.shape[0]+1,1))
            xt[t] = x[t]
            xt[seq_len] = 1
            I_t = np.concatenate((xt,prev_h), axis=0)
            z_t, s_t, h_t = lstm.doForward(W, I_t, prev_s)
            prev_h = h_t
            prev_s = s_t
        pred = Why.dot(h_t)
        loss_i_test = (y-pred)**2/2
        loss_test += loss_i_test
    loss_test = loss_test/float(Y.shape[0])
    print('Epoch: ', epoch+1, 'val_loss: ', loss_test)
    if loss_test<1e-5:
        break
    
print('loss = ' ,loss_test/n_test_data)        
stop = timeit.default_timer() 
print('Time: ', (stop-start)/60, ' minutes')    
images[0].save('pillow_imagedraw.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0) 
# =============================================================================
# MSE of of all the training_data for prediction of T in the future        
# =============================================================================
loss = 0.0     # loss of the prediction of T in the fututre 
phase = np.pi
t = np.arange(0,200,dt)
data = np.sin(2*np.pi*frequency*t+phase)
x = data[0:seq_len*delay:delay]
y = data[seq_len*delay:seq_len*delay+10*1000:delay]
error_max = []
pr = []
#do forward pass
for j in range(y.shape[0]):
    prev_h = np.zeros((hidden_dim, 1))
    prev_s = np.zeros((hidden_dim, 1))
    for t in range(seq_len):
        xt = np.zeros((x.shape[0]+1,1))
        xt[t] = x[t]
        xt[seq_len] = 1
        I_t = np.concatenate((xt,prev_h), axis=0)
        z_t, s_t, h_t = lstm.doForward(W, I_t, prev_s)
        prev_h = h_t
        prev_s = s_t
    pred = Why.dot(h_t)
    pr.append(pred)
    x = np.append(x,pred)
    x = np.delete(x,0)
    loss_i = (y[j]-pred)**2/2
    loss += loss_i
    e = np.abs((y[j]-pred))
    error_max.append(e)
print('mse = ', loss/y.shape[0])
print('maximum absolute error = ', np.amax(error_max))


# =============================================================================
# Get predictions
# =============================================================================
           
#phase = np.pi*.2
#t = np.arange(4,650,dt)
#data = np.sin(2*np.pi*frequency*t+phase)
#x = data[4:4+seq_len*delay:delay]
#x=np.transpose(np.array([x]))
#xx = data[4+seq_len*delay:4+seq_len*delay+10*10000:delay]
#y = data[4:4+seq_len*delay:delay]
#for i in range(x.shape[0]):
#    prev_h = np.zeros((hidden_dim, 1))
#    prev_s = np.zeros((hidden_dim, 1))
#    for t in range(seq_len):
#        xt = np.zeros((x.shape[0]+1,1))
#        xt[t] = x[t]
#        xt[seq_len] = 1
#        I_t = np.concatenate((xt,prev_h), axis=0)
#        z_t, s_t, h_t = lstm.doForward(W, I_t, prev_s)
#        prev_h = h_t
#        prev_s = s_t
#    pred = Why.dot(h_t)
#    x = np.append(x,pred)
#    x = np.delete(x,0)
#    y = np.append(y,pred)
#plt.plot(y)
#plt.plot(data[4:80000:delay], 'r')   

# =============================================================================
# 
# =============================================================================
#loss = 0.0     # loss of the prediction of T in the fututre 
#phase = np.pi
#t = np.arange(0,200,dt)
#data = np.sin(2*np.pi*frequency*t+phase)
#x = data[0:seq_len*delay:delay]
#y = data[seq_len*delay:seq_len*delay+10*100000:delay]
#error_max = []
##do forward pass
#
#j=0
#while (e<.2):
#    prev_h = np.zeros((hidden_dim, 1))
#    prev_s = np.zeros((hidden_dim, 1))
#    for t in range(seq_len):
#        xt = np.zeros((x.shape[0]+1,1))
#        xt[t] = x[t]
#        xt[seq_len] = 1
#        I_t = np.concatenate((xt,prev_h), axis=0)
#        z_t, s_t, h_t = lstm.doForward(W, I_t, prev_s)
#        prev_h = h_t
#        prev_s = s_t
#    pred = Why.dot(h_t)
#    x = np.append(x,pred)
#    x = np.delete(x,0)
#    loss_i = (y[j]-pred)**2/2
#    loss += loss_i
#    e = np.abs((y[j]-pred)/y[j])
#    error_max.append(e)
#    j += 1
#print('t = ', (j+1)*delay*dt)
