#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:53:46 2018

@author: psimin
"""

import eLSH
import KDE
import numpy as np
import mr_hbe as mr
#%%  Problem Specifications


# generate (n,k,d,s)-Instance
kernel = lambda x: np.exp(-x**2)
inverse = lambda mu: np.sqrt(-np.log(mu))
num_points = 1000
clusters = 5
scales = 3
density = 0.001
dimension = 10
spread = 0.02
Instance = KDE.KDE_Instance(kernel,  inverse, num_points, density, 
                 clusters, dimension, scales, spread)
n = Instance.N

eps = 0.5 # multiplicative accuracy of estimation
tau = float(10**-4) # minimum density that we wish to be able to approximate 

#%% Generate hash tables
GLSH_X = eLSH.GLSH(Instance.X, tau, eps)

#%% Unit Test for (n,k,d,s)-Instance
print "================================"
print "Unit Test for random queries"
print "================================"
print "Required relative error < {:.3f}".format(eps) 
print "--------------------------------"

iterations = 30
cnt = 0
for j in range(iterations):
    q = np.zeros(Instance.dimension) + np.random.randn(dimension) / np.sqrt(dimension)
    kernel = lambda x,y: np.exp(-np.linalg.norm(x-y)**2)
    
    kde = 0.0
    for i in range(n):
        kde = kde + kernel(q, Instance.X[:,i])
    kde = kde / n
    est = GLSH_X.AMR(q) 
    print est
    if abs((kde - est) / kde) <= eps:
        cnt = cnt + 1
    print "Query {} rel-error: {:.3f}".format(j+1,(kde - est) / kde)

print "--------------------------------"
print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
print "================================"

#%% Multi-resolution HBE
R = 2*np.sqrt(np.log(1/tau))
mr_gauss = mr.MR_Gauss(tau, Instance.X, R, 0.57, 0.5)
mr_gauss.plot_probabilites()
#%% Unit Test for (n,k,d,s)-Instance
print "================================"
print "Unit Test for random queries"
print "================================"
print "Required relative error < {:.3f}".format(eps) 
print "--------------------------------"

iterations = 20
cnt = 0
for j in range(iterations):
    q = np.zeros(Instance.dimension) + np.random.randn(dimension) / np.sqrt(dimension)
    kernel = lambda x,y: np.exp(-np.linalg.norm(x-y)**2)
    
    kde = 0.0
    for i in range(n):
        kde = kde + kernel(q, Instance.X[:,i])
    kde = kde / n
    est = mr_gauss.AMR(q) 
    print est
    if abs((kde - est) / kde) <= eps:
        cnt = cnt + 1
    print "Query {} rel-error: {:.3f}".format(j+1,(kde - est) / kde)

print "--------------------------------"
print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
print "================================"
