#!/usr/bin/env python
# coding: utf-8

# In[30]:


import time


# In[42]:


type(tic)


# In[49]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression 

tic = time.time()

data = pd.read_csv('ex1data1.txt',header = None)
x = np.array(data.loc[:,0]).reshape(97,1)

y = data.loc[:,1]
#x, X_test,y, y_test = model_selection.train_test_split(x, y, test_size=0.7, random_state=42)
m = x.shape[0]
plt.scatter(x,y)

o = np.ones((m, 1),dtype = int).reshape(m,1)

x = np.concatenate((o, x),axis=1)
y = np.array(y).reshape(m,1)
theta  = np.array([[0 ,0]])
y_predicted = np.dot(theta,x.T)
cost =np.sum(np.square((y_predicted.T - y)))/(97*2)
temp = np.ones(theta.size).reshape(1,2)
alpha = 0.01
print(cost)

for i in range(1000):
    for j in range(theta.shape[1]):
        y_predicted = np.dot(theta,x.T)
        a = y_predicted.T - y
        b = np.array(x[:,j]).reshape(m,1)
        k = np.multiply(a,b)
        temp[0,j] =  theta[0,j] - (alpha/97)*(np.sum(k))
    theta = temp


y_predicted = np.dot(theta,x.T)
cost =np.sum(np.square((y_predicted.T - y)))/(97*2)
print('cost is ',cost)
print('theta is ',theta)


#---------------------------------------------------------------------------


m = x.shape[0]
o = np.ones((m, 1),dtype = int).reshape(m,1)
#X_test = np.concatenate((o, X_test),axis=1)
#y_test = np.array(y_test).reshape(m,1)
y_new =( np.dot(theta,x.T)).T

reg = linear_model.LinearRegression()
reg = reg.fit(x, y)
plt.plot(x[:,1],y_new,color = 'red')
metrics.explained_variance_score(y, y_new)

toc  = time.time()
print((1000*(toc-tic)))


# In[50]:


tic = time.time()

reg = LinearRegression()
reg = reg.fit(x,y)
y_pred = reg.predict(x)
plt.scatter(x[:,1],y)
plt.plot(x[:,1],y_pred,color= 'red')

toc  = time.time()
print((1000*(toc-tic)))


# In[29]:


metrics.explained_variance_score(y,y_pred)


# In[ ]:




