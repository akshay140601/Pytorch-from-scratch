import numpy as np
import matplotlib.pyplot as plt
import random

# TODO: ACCOUNT FOR BIASES!!

class MLP:

    def __init__(self, width, opt_act, opt_init, opt_loss) -> None:
        
        self.width = width
        self.opt_act = opt_act
        self.opt_init = opt_init
        self.weights = self.init_weights()
        self.acts, self.derivs = self.init_activations()
        self.opt_loss = opt_loss

    def activation(self, h, i):

        match self.opt_act[i]:
            case 'sigmoid':
                return (1. / (1. + np.exp(-h)))
            
            case 'relu':
                return np.max(0.0, h)
            
            case 'tanh':
                return ((np.exp(h) - np.exp(-h)) / (np.exp(h) + np.exp(-h)))
                
            case 'linear':
                return h
            
    def derivates(self, h, i):

        match self.opt_act[i]:
            case 'sigmoid':
                #x = self.activation(h)
                return h * (1 - h)
            
            case 'relu':
                if h <= 0:
                    return 0
                else:
                    return 1
                
            case 'tanh':
                #x = self.activation(h)
                return 1 - h**2
            
            case 'linear':
                return 1
            
    def init_weights(self):
        
        match self.opt_init:
            
            case 'xavier':
                weights = []
                np.random.seed(0)
                for i in range(len(self.width) - 1):
                    var = 1. / self.width[i]
                    w_i = np.random.normal(0, np.sqrt(var), (self.width[i], self.width[i + 1]))
                    weights.append(w_i)

                print('Xavier weights: ')
                print(weights)
                return weights
            
            case 'he':
                weights = []
                np.random.seed(0)
                for i in range(len(self.width) - 1):
                    var = 2. / self.width[i]
                    w_i = np.random.normal(0, np.sqrt(var), (self.width[i], self.width[i + 1]))
                    weights.append(w_i)

                return weights
            
    def init_activations(self):

        acts = []
        derivs = []
        for i in range(len(self.width) - 1):
            a = np.random.rand(self.width[i], self.width[i + 1])
            d = np.random.rand(self.width[i], self.width[i + 1])
            acts.append(a)
            derivs.append(d)

        return acts, derivs
    
    def forward(self, input_data):
        
        acts = input_data
        for i in range(len(self.width) - 1):
            pre_acts = np.dot(acts, self.weights[i])
            acts = self.activation(pre_acts, i)  #TODO: Don't put sigmoid to last layer
            self.acts[i] = acts
        print(self.weights)
        print(self.acts)

        return self.acts
    
    def loss(self):
        
        match self.opt_loss:
            case 'l2':
                pass

            case 'ce':
                pass
    
    def backward(self, de):
        
        error_de = de
        for i in reversed(range(len(self.width) - 1)):
            dL = error_de * self.derivates(self.acts[i], i)
            self.derivs[i] = np.dot(self.acts[i - 1], dL)
            error_de = np.dot(dL, self.weights[i].T)



if __name__ == '__main__':
    X = np.array([2,1,3])
    activation = 'Linear'
    width = [3,4,2]
    n_layers = 3

    mlp = MLP([3,4,2], 'sigmoid', 'xavier')
    mlp.forward(X)
