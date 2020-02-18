#!/usr/bin/env python
# coding: utf-8

# In[898]:


# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

import pandas as pd


# In[899]:


data = pd.read_csv("house_data_complete.csv").dropna()
data


# In[900]:


x,y = data["bedrooms"],data["price"]
lambda_ = 1
priceList = []
bedroomList = []
priceList = list(y)
bedroomList = list(x)

pyplot.plot(bedroomList, priceList, 'ro', ms=10, mec='k')
pyplot.ylabel('Price')
pyplot.xlabel('Bedrooms')


# In[901]:


columns = data.columns[3:]
norm_data = (data[columns] - (data[columns]).mean())/(data[columns]).std()
print(norm_data)
data[columns] = norm_data


# In[902]:


train, validate, test = np.split(data.sample(frac=1),[int(.6*len(data)),int(.8*len(data))])


# In[903]:


train


# In[904]:


train = train.to_numpy()
train


# In[905]:


validate = validate.to_numpy()
validate


# In[906]:


test = test.to_numpy()
test


# In[907]:


train_x = train[:, [3, 4]]
train_x


# In[908]:


train_y = train[:,2]
train_y


# In[909]:


train_m = train_y.size
train_m


# In[910]:


train_x = np.concatenate([np.ones((train_m,1)), train_x], axis=1)
train_x


# In[911]:


def computeCostMulti(train_x, train_y, theta, lambda_):
    
    # Initialize some useful values
    train_m = train_y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    
    J = 0
    J = np.dot((np.dot(train_x,theta)-train_y),(np.dot(train_x,theta)-train_y))/(2*train_m)+((lambda_/(2*train_m))*np.sum(np.dot(theta,theta)))
    
    # ==================================================================
    return J


# In[912]:


def gradientDescentMulti(train_x, train_y, theta, alpha, num_iters,lambda_):
    
    # Initialize some useful values
    train_m = train_y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        sumofh0x=np.dot(train_x,theta)
        theta = theta*(1-(alpha*lambda_)/train_m)-((alpha/train_m)*(np.dot(train_x.T,sumofh0x-train_y)))
        #theta=theta-((alpha/train_m)*(np.dot(train_x.T,sumofh0x-train_y)))
        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(train_x, train_y, theta,lambda_))
    
    return theta, J_history


# In[913]:


# Choose some alpha value - change this
alpha = 0.1
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(train_x, train_y, theta, alpha, num_iters,lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[914]:


def computeCostMulti2(train_x, train_y, theta,lambda_):
    
    # Initialize some useful values
    train_m = train_y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    
    J = 0
    J = np.dot((np.dot(np.square(train_x),theta)-train_y),(np.dot(np.square(train_x),theta)-train_y))/(2*train_m)+((lambda_/(2*train_m))*np.sum(np.dot(theta,theta)))
    
    # ==================================================================
    return J


# In[915]:


def gradientDescentMulti2(train_x, train_y, theta, alpha, num_iters, lambda_):
    
    # Initialize some useful values
    train_m = train_y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        sumofh0x=np.dot(np.square(train_x),theta)
        theta = theta*(1-(alpha*lambda_)/train_m)-((alpha/train_m)*(np.dot(train_x.T,sumofh0x-train_y)))
        #theta=theta-((alpha/train_m)*(np.dot(np.square(train_x).T,sumofh0x-train_y)))
        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti2(train_x, train_y, theta,lambda_))
    
    return theta, J_history


# In[916]:


# Choose some alpha value - change this
alpha = 0.0082
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti2(train_x, train_y, theta, alpha, num_iters,lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[917]:


def computeCostMulti3(train_x, train_y, theta,lambda_):
    
    # Initialize some useful values
    train_m = train_y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    # ======================= YOUR CODE HERE ===========================
    
    J = 0
    J = np.dot((np.dot(np.power(train_x,3),theta)-train_y),(np.dot(np.power(train_x,3),theta)-train_y))/(2*train_m)+((lambda_/(2*train_m))*np.sum(np.dot(theta,theta)))
    
    # ==================================================================
    return J


# In[918]:


def gradientDescentMulti3(train_x, train_y, theta, alpha, num_iters,lambda_):
    
    # Initialize some useful values
    train_m = train_y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        sumofh0x=np.dot(np.power(train_x,3),theta)
        theta=theta*(1-(alpha*lambda_)/train_m)-((alpha/train_m)*(np.dot(train_x.T, sumofh0x-train_y)))
        #theta=theta-((alpha/train_m)*(np.dot(np.power(train_x,3).T,sumofh0x-train_y)))
        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti3(train_x, train_y, theta,lambda_))
    
    return theta, J_history


# In[919]:


# Choose some alpha value - change this
alpha = 0.014
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti3(train_x, train_y, theta, alpha, num_iters,lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[920]:


validate_x = validate[:, [3, 4]]
validate_x


# In[921]:


validate_y = validate[:,2]
validate_y


# In[922]:


validate_m = validate_y.size
validate_m


# In[923]:


validate_x = np.concatenate([np.ones((validate_m,1)), validate_x], axis=1)
validate_x


# In[924]:


# Choose some alpha value - change this
alpha = 0.1 # as alpha increases to 0.1, curve converges sooner
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(validate_x, validate_y, theta, alpha, num_iters, lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[925]:


# Choose some alpha value - change this
alpha = 0.0082
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti2(validate_x, validate_y, theta, alpha, num_iters, lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[926]:


# Choose some alpha value - change this
alpha = 0.014 
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti3(validate_x, validate_y, theta, alpha, num_iters, lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[927]:


test_x = test[:, [3, 4]]
test_x


# In[928]:


test_y = test[:,2]
test_y


# In[929]:


test_m = test_y.size
test_m


# In[930]:


test_x = np.concatenate([np.ones((test_m,1)), test_x], axis=1)
test_x


# In[931]:


# Choose some alpha value - change this
alpha = 0.1 # as alpha increases to 0.1, curve converges sooner
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(test_x, test_y, theta, alpha, num_iters, lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[932]:


# Choose some alpha value - change this
alpha = 0.0082
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti2(test_x, test_y, theta, alpha, num_iters, lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[933]:


# Choose some alpha value - change this
alpha = 0.014
num_iters = 100

# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti3(test_x, test_y, theta, alpha, num_iters, lambda_)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')


# In[ ]:




