#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 03:23:44 2020

@author: bda-lab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn import matrices
from sklearn import model_selection



data = pd.read_csv('ex1data1.txt',header = None)
x = np.array(data.loc[:,0]).reshape(97,1)

y = data.loc[:,1]
x, X_test,y, y_test = model_selection.train_test_split(x, y, test_size=0.33, random_state=42)
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


for i in range(10):
    print(theta)
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