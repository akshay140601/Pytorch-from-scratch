import numpy as np
import matplotlib.pyplot as plt
import random
from numpyNN import *
from MLP import *

X_train, y_train, X_test, y_test = sample_data('linear-separable', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 1, 1], ['linear', 'sigmoid', 'linear'], 'xavier', 'l2', 'vanilla_gd', 0.01, 0, 0, 0, 0)
mlp.train_and_test(100, X_train, y_train, X_test, y_test)