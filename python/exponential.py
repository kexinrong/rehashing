#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:12:53 2018

@author: psimin
@email: psiminelakis@gmail.com

This is an implementation of the methodology of Hashing Based Estimators
for kernel density estimation under the Exponential kernel.
"""

import eLSH
import KDE
import numpy as np

def estimate_diameter(X):
    """
    Estimate the diameter of the set by 2 * maximum norm of points
    i.e. using 0 as the ``center"
    """
    n = X.shape[0]
    radius = 0
    for i in range(n):
        if np.linalg.norm(X[:,i]) > radius:
            radius = np.linalg.norm(X[:,i])
    return 2 * radius

def set_width(diameter):
    """
    This is the width of the hash buckets used to define the LSH
    """
    
    return 6 * np.ceil(diameter**2) / np.sqrt(np.pi/2)

def set_power(diameter):
    """
    This is the number of independent hash-functions concatenated to create
    a hash function with the desirable hashing probability
    """
    
    return int(3 * np.ceil(diameter**2))

#%%  Problem Specifications
n = 100 # number of points
d = 8  # dimension
eps = 0.5 # multiplicative accuracy of estimation
tau = float(1.0/n) # minimum density that we wish to be able to approximate 
delta = 1.0 / n # bound on the probability of failure
RelVar = lambda mu: np.e**1.5 / np.sqrt(mu) # Bound on the relative variance
means = np.ceil(6 * RelVar(tau) / eps**2) # Number of means required
meds = np.ceil(9 * np.log(1.0 / delta)) # number of medians to get low-prob-err
M = int(means * meds)

# generate random dataset
X = np.random.randn(d, n)/d 
#%% Set Parameters
diam = estimate_diameter(X)
w = set_width(diam)
k = set_power(diam)

#%% Generate hash tables
e2lsh_X = eLSH.E2LSH(M, X, w, k)

#%% Unit Test for Random Queries Acc 0.5
eps = 0.5
print "================================"
print "Unit Test for random queries"
print "================================"
print "Required relative error < {:.3f}".format(eps)
print "Required failure prob < {:.3f}".format(delta)
print "--------------------------------"
iterations = 10
cnt = 0
for j in range(iterations):
    # q = np.random.randn(d) # worst case
    q = np.random.randn(d)/d
    kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))
    
    kde = 0.0
    for i in range(n):
        kde = kde + kernel(q, X[:,i])
    kde = kde / n
    est = e2lsh_X.query(q, eps, delta, tau) 
    if abs((kde - est) / kde) <= eps:
        cnt = cnt + 1
    print "Query {} rel-error: {:.3f}".format(j+1,(kde - est) / kde)

print "--------------------------------"
print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
print "================================"
#%% Unit Test for Random Queries Acc 0.8
eps = 0.8
print "================================"
print "Unit Test for random queries"
print "================================"
print "Required relative error < {:.3f}".format(eps)
print "Required failure prob < {:.3f}".format(delta)
print "--------------------------------"
iterations = 30
cnt = 0
for j in range(iterations):
    q = np.random.randn(d)
    kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))
    
    kde = 0.0
    for i in range(n):
        kde = kde + kernel(q, X[:,i])
    kde = kde / n
    est = e2lsh_X.query(q, eps, delta, tau) 
    if abs((kde - est) / kde) <= eps:
        cnt = cnt + 1
    print "Query {} rel-error: {:.3f}".format(j+1,(kde - est) / kde)

print "--------------------------------"
print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
print "================================"

#%%  Problem Specifications


# generate (n,k,d,s)-Instance
kernel = lambda x: np.exp(-x)
inverse = lambda mu: -np.log(mu)
num_points = 10
clusters = 5
scales = 4
density = 1.0 / (num_points * clusters * scales)
dimension = 10
spread = 0.02
Instance = KDE_Instance(kernel,  inverse, num_points, density, 
                 clusters, dimension, scales, spread)
n = Instance.N

eps = 1.0 # multiplicative accuracy of estimation
tau = float(1.0/n) # minimum density that we wish to be able to approximate 
delta = 1.0 / np.e # bound on the probability of failure
RelVar = lambda mu: np.e**1.5 / np.sqrt(mu) # Bound on the relative variance
means = np.ceil(6 * RelVar(tau) / eps**2) # Number of means required
meds = np.ceil(9 * np.log(1.0 / delta)) # number of medians to get low-prob-err
M = int(means * meds)
#%% Set Parameters
diam = estimate_diameter(Instance.X)
w = set_width(diam)
k = set_power(diam)

#%% Generate hash tables
e2lsh_X = eLSH.E2LSH(M, Instance.X, w, k)

#%% Unit Test for (n,k,d,s)-Instance
eps = 0.8
print "================================"
print "Unit Test for random queries"
print "================================"
print "Required relative error < {:.3f}".format(eps)
print "Required failure prob < {:.3f}".format(delta)
print "--------------------------------"

iterations = 30
cnt = 0
for j in range(iterations):
    q = np.zeros(Instance.dimension)
    kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))
    
    kde = 0.0
    for i in range(n):
        kde = kde + kernel(q, Instance.X[:,i])
    kde = kde / n
    est = e2lsh_X.query(q, eps, delta, tau) 
    if abs((kde - est) / kde) <= eps:
        cnt = cnt + 1
    print "Query {} rel-error: {:.3f}".format(j+1,(kde - est) / kde)

print "--------------------------------"
print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
print "================================"


