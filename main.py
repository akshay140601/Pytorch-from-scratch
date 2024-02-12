#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random
from numpyNN import *
from MLP import *


# In[2]:


# Deliverable 2 - Linearly separable dataset

X_train, y_train, X_test, y_test = sample_data('linear-separable', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 1, 1], ['linear', 'sigmoid', 'linear'], 'xavier', 'l2', 'vanilla_gd', 0.001, 0, 0, 0, 0)
mlp.train_and_test(800, X_train, y_train, X_test, y_test)


# In[3]:


# Deliverable 3 - XOR Dataset

X_train, y_train, X_test, y_test = sample_data('XOR', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50 ,30, 20, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'he', 'ce', 'adam', 0.001, 1, 0.9, 0.999, 1e-8)
mlp.train_and_test(200, X_train, y_train, X_test, y_test)


# In[4]:


# Deliverable 4 - Circle Dataset - CE Loss Function

X_train, y_train, X_test, y_test = sample_data('circle', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50, 40, 30, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'ce', 'adam', 0.001, 1, 0.9, 0.999, 1e-8)
mlp.train_and_test(750, X_train, y_train, X_test, y_test)


# In[5]:


# Deliverable 4 - Circle Dataset - L2 loss function

X_train, y_train, X_test, y_test = sample_data('circle', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50, 40, 30, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'l2', 'adam', 0.001, 1, 0.9, 0.999, 1e-8)
mlp.train_and_test(500, X_train, y_train, X_test, y_test)


# In[24]:


# Deliverable 5 - Sinusoid dataset - Adam Optimizer

X_train, y_train, X_test, y_test = sample_data('sinusoid', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50, 40, 30, 20, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'ce', 'adam', 0.001, 1, 0.9, 0.999, 1e-8)
mlp.train_and_test(1750, X_train, y_train, X_test, y_test)


# In[7]:


# Deliverable 5 - Sinusoid dataset - Gradient Descent with Momentum Optimizer

X_train, y_train, X_test, y_test = sample_data('sinusoid', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50, 40, 30, 20, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'ce', 'gd_with_momentum', 0.001, 0.9, 0.9, 0.999, 1e-8)
mlp.train_and_test(2000, X_train, y_train, X_test, y_test)


# In[8]:


# Deliverable 5 - Sinusoid dataset - Vanilla Gradient Descent Optimizer

X_train, y_train, X_test, y_test = sample_data('sinusoid', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50, 40, 30, 20, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'ce', 'vanilla_gd', 0.001, 0.9, 0.9, 0.999, 1e-8)
mlp.train_and_test(2000, X_train, y_train, X_test, y_test)


# In[10]:


# Deliverable 6 - Swiss roll dataset

X_train, y_train, X_test, y_test = sample_data('swiss-roll', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
mlp = MLP([X_train.shape[1], 50, 40, 30, 20, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'ce', 'adam', 0.001, 1, 0.9, 0.999, 1e-8)
mlp.train_and_test(1500, X_train, y_train, X_test, y_test)


# In[2]:


# Deliverable 7 - Non-linear embeddings - Circle dataset

X_train, y_train, X_test, y_test = sample_data('circle', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train_non_linear = np.sum(X_train ** 2, axis = 1)
X_test_non_linear = np.sum(X_test ** 2, axis = 1)
X_train = np.column_stack((X_train, X_train_non_linear))
X_test = np.column_stack((X_test, X_test_non_linear))


# In[7]:


mlp = MLP([X_train.shape[1], 50, 40, 30, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'ce', 'adam', 0.001, 1, 0.9, 0.999, 1e-8, non_linear = True)
mlp.train_and_test(1200, X_train, y_train, X_test, y_test)


# In[8]:


# Deliverable 7 - Non-linear embeddings - XOR dataset

X_train, y_train, X_test, y_test = sample_data('XOR', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train_non_linear = np.sum(X_train ** 2, axis = 1)
X_test_non_linear = np.sum(X_test ** 2, axis = 1)
X_train = np.column_stack((X_train, X_train_non_linear))
X_test = np.column_stack((X_test, X_test_non_linear))


# In[16]:


mlp = MLP([X_train.shape[1], 8, 4, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'he', 'ce', 'adam', 0.001, 1, 0.9, 0.999, 1e-8, non_linear = True)
mlp.train_and_test(1500, X_train, y_train, X_test, y_test)


# In[17]:


# Deliverable 7 - Non-linear embeddings - Swiss-roll dataset

X_train, y_train, X_test, y_test = sample_data('swiss-roll', 200, 200, 0)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train_non_linear = np.sum(X_train ** 2, axis = 1)
X_test_non_linear = np.sum(X_test ** 2, axis = 1)
X_train = np.column_stack((X_train, X_train_non_linear))
X_test = np.column_stack((X_test, X_test_non_linear))


# In[22]:


mlp = MLP([X_train.shape[1], 50, 40, 30, 20, 1], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], 'xavier', 'l2', 'adam', 0.001, 1, 0.9, 0.999, 1e-8, non_linear = True)
mlp.train_and_test(1500, X_train, y_train, X_test, y_test)


# In[ ]:




