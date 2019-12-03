  # -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 13:05:10 2019

@author: barzegar
"""

import numpy as np

'''define activation functions '''

class Linear:
    def forward(self, x): return x
    def backward(self, dy): return dy
    
    
class ReLU:
    def forward(self, x):
        y = np.max(0, x)
        self.x = x
        return y
    
    def backward(self, y):
        dy = (self.x>0)
        return dy
    
class tanh:
    def forward(self, x):
        y = np.tanh(x)
        self.x = x
        return y
    
    def backwards(self, y):
        dy = 1-np.tanh(y)**2
        return dy
    
class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backwards(self, y):
        s = 1 / (1 + np.exp(-y))
        dy = s*(1-s)
        return dy
    
class LSTM:
    ''' one LSTM cell and one time step'''
    def doForward(self, W, I, prev_s, NL=tanh()):
        z_hat = W.dot(I)
        a = Sigmoid()
        l = int(W.shape[0]/4)
        g_t = NL.forward(z_hat[0:l])
        i_t = a.forward(z_hat[l:l*2])
        f_t = a.forward(z_hat[2*l:3*l])
        o_t = a.forward(z_hat[3*l:4*l])
        s_t = np.multiply(i_t,g_t) + np.multiply(prev_s, f_t)
        h_t = np.multiply(o_t, NL.forward(s_t))
        z_t = [g_t, i_t, f_t, o_t]
        return z_t, s_t, h_t,
    
    def doBackwards(self, dh, dprev_s, forward, I, NL=tanh()):
        s_t = forward[0]
        prev_s = forward[1]
        g_t = forward[2]
        i_t = forward[3]
        f_t = forward[4]
        o_t = forward[5]
        do = np.multiply(dh, NL.forward(s_t))
        dh_s = np.multiply(o_t, NL.backwards(s_t))
        ds = np.multiply(dh, dh_s) + dprev_s
        di = np.multiply(ds, g_t)
        dprev_s = ds
#        print(ds)
        dg = np.multiply(ds, i_t)
        df = np.multiply(ds, prev_s)
        dg_hat = np.multiply(dg, NL.backwards(g_t))
        di_hat = np.multiply(di, NL.backwards(i_t))
        df_hat = np.multiply(df, NL.backwards(f_t))
        do_hat = np.multiply(do, NL.backwards(o_t))
#        print(do_hat)
        dz = np.concatenate((dg_hat, di_hat, df_hat, do_hat), axis=0)
        dW = dz.dot(I.T)
        
        return dprev_s, dW
        
        
class Layer:
    '''do forward and backward pass for a layer '''  
    def __init__(self, nInput, hidden_dim):
        self.hidden_dim = hidden_dim
        self.seq_len = nInput
        
    def setRandomWeights(self):
        Wxh = np.random.normal(0, 1, (self.hidden_dim, self.seq_len+1))
        Whh = np.random.normal(0, 1, (self.hidden_dim, self.hidden_dim))
        W1 = np.concatenate((Wxh, Wxh, Wxh, Wxh),axis=0)
        W2 = np.concatenate((Whh, Whh, Whh, Whh), axis=0)
        self.W = np.concatenate((W1, W2), axis=1)
        self.Why = np.random.normal(0, 1, (self.output_dim, self.hidden_dim))
        
#    def setHiddenToOutputRandomWeights(self):
#        self.Why = np.random.normal(0, 1, (self.output_dim, self.hidden_dim))
        
    def doForward(self, _input):
        self.x = _input
        self.remember_hidden = []
        self.remember_z = []
        _output = []
        prev_h = np.zeros((self.hidden_dim, 1))
        prev_s = np.zeros((self.hidden_dim, 1))
        lstm = LSTM()
        for t in range(self.seq_len):
            xt = np.zeros((self.x.shape[0]+1,1))
            xt[t] = self.x[t]
            xt[self.seq_len] = 1
            I_t = np.concatenate((xt,prev_h), axis=0)
            z_t, s_t, h_t = lstm.doForward(self.W, I_t, prev_s)
            self.remember_hidden.append({'h':h_t, 'prev_h':prev_h, 's':s_t, 'prev_s':prev_s})
            self.remember_z.append({'zt':z_t})
            _output.append(h_t)
            prev_h = h_t
            prev_s = s_t
        _output = np.hstack(_output)
        prediction = self.Why.dot(h_t)    
        return prediction, _output
            
    def doBackwards(self, dOutput):
        
        return
    
    def updateWeights(self, learning_rate = 0.001):
        self.Why -= learning_rate*self.dWhy
        self.W -= learning_rate*self.dW
        return self.Why, self.W
    
class RecurrentNetwork:
    '''a recurrent neural network '''
    def __init__(self, layers):
        '''layers = ( (nInputs, nOutputs, hidden_dim), ...) '''
        self.nLayers = len(layers)
        self.I = [None]*(self.nLayers+1)
        self.dI = [None]*(self.nLayers+1)
        self.layers = [ Layer(layers[0][0], layers[0][1]) ]
        for l in range(1,self.nLayers):
            self.layers.append( Layer(layers[l][0], layers[l][1]) )
        self.setRandomWeights()
        
    def setRandomWeights(self):
        for l in range(0,self.nLayers):
            self.layers[l].setRandomWeights()
#        self.layers[self.nLayers].setHiddenToOutputRandomWeights()
        
    def doForward(self, _input):
        self.I[0] = _input
        for l in range(self.nLayers):
            prediction, self.I[l+1] = self.layers[l].doForward(self.I[l])
        return prediction, self.I[self.nLayers]
        
class ObjectiveFunction:
    '''loss function definition '''
    def mseForward(self, pred, y):
        J = (y-pred)**2/2
        return J
