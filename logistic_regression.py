#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection


# In[38]:


data = datasets.load_iris()
X = data.data
y = data.target
theta = np.array([0,0,0,0])


# In[48]:


z = np.dot(theta,X.T)
y_predicted  = sigmoid(z)


# In[73]:


def sigmoid(z):
    minus_z = np.multiply(-1,z)
    temp = np.exp(minus_z)
    k = np.add(1,z)
    y_predicted = np.divide(1,k)
    print(y_predicted)
    return(y_predicted)
    


# In[76]:


def cost(y_predicted,y):
    log_y = np.log(y_predicted)
    a = (-1)*(np.dot(y,log_y))
    b = np.dot(np.subtract(1,y),np.log(np.subtract(1,y_predicted)))
    cost = np.subtract(a,b)
    return(cost)


# In[77]:


print(cost(y_predicted,y))


# In[ ]:




