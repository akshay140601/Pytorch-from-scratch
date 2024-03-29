import numpy as np
import matplotlib.pyplot as plt
import random
from numpyNN import plot_loss

# TODO: ACCOUNT FOR BIASES!!

class MLP:

    def __init__(self, width, opt_act, opt_init, opt_loss, optimizer, learning_rate, momentum, beta1, beta2, epsilon, non_linear = False) -> None:
        
        self.width = width
        self.opt_act = opt_act
        self.opt_init = opt_init
        self.weights, self.derivs = self.init_weights()
        self.acts = self.init_activations()
        self.opt_loss = opt_loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = self.init_optimizer_gd_momentum()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m, self.v, self.m_hat, self.v_hat, self.t = self.init_optimizer_adam()
        self.non_linear = non_linear

    def activation(self, h, i):

        match self.opt_act[i]:
            case 'sigmoid':
                return (1. / (1. + np.exp(-h)))
            
            case 'relu':
                return np.maximum(0.0, h)
            
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
                return np.where(h <= 0, 0, 1)
                
            case 'tanh':
                #x = self.activation(h)
                return 1 - h**2
            
            case 'linear':
                return 1
            
    def init_weights(self):
        
        match self.opt_init:
            
            case 'xavier':
                weights = []
                derivs = []
                np.random.seed(0)
                for i in range(len(self.width) - 1):
                    var = 1. / self.width[i]
                    w_i = np.random.normal(0, np.sqrt(var), (self.width[i + 1], self.width[i]))
                    d = np.random.rand(self.width[i + 1], self.width[i])
                    weights.append(w_i)
                    derivs.append(d)

                return weights, derivs
            
            case 'he':
                weights = []
                derivs = []
                np.random.seed(0)
                for i in range(len(self.width) - 1):
                    var = 2. / self.width[i]
                    w_i = np.random.normal(0, np.sqrt(var), (self.width[i + 1], self.width[i]))
                    weights.append(w_i)
                    d = np.random.rand(self.width[i + 1], self.width[i])
                    derivs.append(d)

                return weights, derivs
            
    def init_activations(self):

        acts = []
        for i in range(len(self.width)):
            a = np.random.rand(self.width[i])
            acts.append(a)

        return acts
    
    def forward(self, input_data):
        
        acts = input_data
        self.acts[0] = input_data
        j = 1
        for i in range(len(self.width) - 1):
            pre_acts = self.weights[i] @ acts
            #pre_acts = np.dot(acts, self.weights[i])
            #print('Before Activation: ', pre_acts)
            acts = self.activation(pre_acts, i)  #TODO: Don't put sigmoid to last layer
            #print('After activation: ', acts)
            self.acts[j] = acts
            j += 1
        #print(self.weights)
        #print(self.acts)
    
    def loss(self, predicted, true):
        
        match self.opt_loss:
            case 'l2':
                l = (true - predicted) ** 2
                l_deriv = -2 * (true - predicted)

                return l, l_deriv

            case 'ce':
                '''l = - ((true * np.log(predicted)) + (1 - true) * np.log(1 - predicted))
                l_deriv = (predicted - true) / (predicted * (1 - predicted))'''
                #l = -np.mean(true * np.log(predicted) + (1 - true) * np.log(1 - predicted))
                #l_deriv = (predicted - true) / (predicted * (1 - predicted) + 1e-8) / len(true)  # Adding a small epsilon to avoid division by zero
                epsilon = 1e-6
                #predicted = np.clip(predicted, epsilon, 1 - epsilon)
                ce_loss = - (true * np.log(predicted) + (1 - true) * np.log(1 - predicted))
                l = np.sum(ce_loss)
                l_deriv = (predicted - true) / (predicted * (1 - predicted))
                return l, l_deriv


                #return l, l_deriv
    
    def backward(self, error_de):
        
        #l, l_deriv = self.loss(self.acts[-1], np.array([0.5]))
        #error_de = l_deriv
        error_de = error_de.reshape(-1, 1)
        #print(self.derivs)
        for i in reversed(range(len(self.width) - 1)):
            act_deriv = self.derivates(self.acts[i + 1], i)
            if (type(act_deriv) == int):
                act_deriv = np.array([act_deriv]).reshape(-1, 1)
            else:
                act_deriv = act_deriv.reshape(-1, 1)
            dL = error_de * act_deriv
            prev_layer_acts = self.acts[i].reshape(-1, 1)
            self.derivs[i] = dL @ prev_layer_acts.T
            error_de = self.weights[i].T @ dL

        #print(self.derivs)
    
    
    def init_optimizer_adam(self):

        m = []
        v = []
        m_hat = []
        v_hat = []
        t = 0
        for i in range(len(self.width) - 1):
            M = np.zeros((self.width[i + 1], self.width[i]))
            m.append(M)
            v.append(M)
            m_hat.append(M)
            v_hat.append(M)
        
        return m, v, m_hat, v_hat, t
    
    def init_optimizer_gd_momentum(self):

        velocity = []
        for i in range(len(self.width) - 1):
            v = np.zeros((self.width[i + 1], self.width[i]))
            velocity.append(v)
        return velocity

    def optimizer_step(self):
        
        match self.optimizer:
            case 'vanilla_gd':
                
                for i in range(len(self.width) - 1):
                    self.weights[i] -= self.learning_rate * self.derivs[i]

            case 'gd_with_momentum':
                
                for i in range(len(self.width) - 1):
                    self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * self.derivs[i]
                    self.weights[i] -= self.velocity[i]


            case 'adam':

                self.t += 1
                for i in range(len(self.width) - 1):
                    self.m[i] = (self.beta1 * self.m[i]) + (1 - self.beta1) * self.derivs[i]
                    self.v[i] = (self.beta2 * self.v[i]) + (1 - self.beta2) * (self.derivs[i] * self.derivs[i])
                    self.m_hat[i] = self.m[i] / (1 - self.beta1 ** self.t)
                    self.v_hat[i] = self.v[i] / (1 - self.beta2 ** self.t)
                    self.weights[i] -= (self.learning_rate * self.m_hat[i]) / (np.sqrt(self.v_hat[i]) + self.epsilon)

    def plot_decision_boundary(self, X, y, boundry_level=None):
        """
        Plots the decision boundary for the model prediction
        :param X: input data
        :param y: true labels
        :param pred_fn: prediction function,  which use the current model to predict。. i.e. y_pred = pred_fn(X)
        :boundry_level: Determines the number and positions of the contour lines / regions.
        :return:
        """
        
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
        input_data = np.c_[xx.ravel(), yy.ravel()]

        Z = np.zeros((input_data.shape[0], 1))
        
        for i in range(input_data.shape[0]):
            self.forward(input_data[i])
            Z[i, 0] = self.acts[-1]

        #Z = pred_fn(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r',)
        plt.show()

    def plot_decision_boundary_non_linear_embeddings(self, X, y, boundry_level=None):
        """
        Plots the decision boundary for the model prediction
        :param X: input data
        :param y: true labels
        :param pred_fn: prediction function,  which use the current model to predict。. i.e. y_pred = pred_fn(X)
        :boundry_level: Determines the number and positions of the contour lines / regions.
        :return:
        """
        
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        xx3 = (xx ** 2 + yy ** 2).reshape(xx.shape)
        
        input_data = np.c_[xx.ravel(), yy.ravel(), xx3.ravel()]

        Z = np.zeros((input_data.shape[0], 1))
        
        for i in range(input_data.shape[0]):
            self.forward(input_data[i])
            Z[i, 0] = self.acts[-1]

        #Z = pred_fn(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r',)
        plt.show()

    
    def train_and_test(self, epochs, X_train, y_train, X_test, y_test):

        losses = {'train_loss': [], 'test_loss': []}
        
        for epoch in range(epochs):

            train_loss = []
            test_loss = []

            for i in range(len(X_train)):
                self.forward(X_train[i])
                #print(self.acts[-1])

                l, l_deriv = self.loss(self.acts[-1], y_train[i])

                self.backward(l_deriv)

                self.optimizer_step()

                train_loss.append(l)

                self.forward(X_test[i])
                l, _ = self.loss(self.acts[-1], y_test[i])

                test_loss.append(l)

            losses['train_loss'].append(np.mean(train_loss))
            losses['test_loss'].append(np.mean(test_loss))

        plot_loss(losses)
        if self.non_linear == False:
            self.plot_decision_boundary(X_test, y_test)
        else:
            self.plot_decision_boundary_non_linear_embeddings(X_test, y_test)




if __name__ == '__main__':
    X = np.array([2,1,3])
    width = [3,4,1]
    n_layers = 3

    mlp = MLP([3,4,2], ['linear', 'sigmoid', 'linear'], 'xavier', 'ce', 'gd', 0.01, 0, 0, 0, 0, 0)
    mlp.forward(X)
    mlp.backward()
