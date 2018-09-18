#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:02:31 2017

@author: paris
@email: psiminelakis@gmail.com

T
"""
import numpy as np
from scipy.special import erf as erf

def hashfunction(G, w, b):
    """
    Randomly project on k-dimensions, add a random shift,
    discretize each dimension by buckets of width w
    """
    
    return lambda x: tuple(np.ceil((G.dot(x) + w * b) / w))

class hash_bucket:
    
    def __init__(self, point):
        """
        keep a single point from each bucket as well as
        the total number of points
        """
        
        self.point = point
        self.num = 1
    
    
    def update(self, point):
        """
        Use reservoir sampling to keep only a single uniform random point
        """
        
        self.num = self.num + 1 #number of points seens so far
        if np.random.rand() <= 1.0 / self.num: # reservoir sampling
            self.point = point # update point
            
class hash_table:
    
    
    def __init__(self):
        self.H = {}
    
    def build(self, X, w, k):
        """
        Build Hash table by:
            - Sampling a hash function from E2LSH
            - Evaluating the hash function on the data set
        """
        
        # Set parameters of hash function
        self.scale = w
        self.k = k
        # sample hash function from E2LSH
        d = X.shape[0]
        G = np.random.randn(k, d)
        b = np.random.rand(k)
        self.h = hashfunction(G, w, b) 
        
        # evaluate hash function on data set X
        n = X.shape[1]
        for i in range(n):
            key = self.h(X[:, i])
            if not (self.H).has_key(key):
                self.H[key] = hash_bucket(X[:, i]) # create bucket
            else:
                self.H[key].update(X[:, i]) # update bucket
                
    def sample(self, q):
        """
        Search hash table for bucket where query maps to, if empty return False
        """
        key = self.h(q)
        if (self.H).has_key(key): # if bucket not empty
            return self.H[key].point # return point
        else:
            return False # bucket is empty

class ELSH:
    
    def __init__(self, M, X, w, k):
        
        self.HT = [hash_table() for i in range(M)]
        self.scale = w # scale to normalize distances
        self.num = M # number of Hash tables / samples
        self.cur = 0 # current index of Hash Table to query
        self.power = k # concatentation of k hash functions 
        self.N = X.shape[1] #number of points
        for i in range(M):
            self.HT[i].build(X, w, k)
        print('built {} hash functions'.format(M))    
    
    def prob(self, c):
        """
        returns the collision probability of two points with normalized dist, c
        """
        
        return (erf(1.0 / c) - np.sqrt(2.0 / np.pi)*\
                c * (1 - np.exp(- 1.0 / (2 * c**2))))**self.power
    
    def kernel(self, x,y):
        """
        Gaussian kernel exp(-||x-y||^2)
        """
        
        return np.exp(-np.linalg.norm(x-y)**2)
    
    def evalquery(self, q):
        """
        cycle through hash tables and computes HBE(q)
        """
        
        # find current index
        if self.cur < self.num:
            i = self.cur
        else:
            i = 0
        self.cur = i +1
        
        # Sample a point from the hash bucket
        Xi = self.HT[i].sample(q)
        if not np.array(Xi).any(): 
            return 0.0 # return 0 if empty
        else:
            key = self.HT[i].h(q) # hash query
            c = np.linalg.norm(q - Xi) / self.scale # compute normalized dist
            return self.kernel(Xi, q) / self.prob(c) *\
                        self.HT[i].H[key].num / self.N # return HBE

class GLSH:
        
    def __init__(self, X, tau, eps):
        self. eps = eps
        self.R = np.sqrt(np.log(1 / tau)) # effective diameter of the set
        self.gamma = 0.5           # rate that we decrease our estimate
        self.I = int(np.ceil(np.log2(1 / tau))) # number of different guesses
        print(self.I)
        self.mui = np.array([(1 - self.gamma) ** i for i in range(self.I)])  
        self.ti = np.sqrt(np.log(1/self.mui)) / 2.0 # nominal scale for level i
        self.ki = [int(3 * np.ceil(self.R * self.ti[j]))  
                    for j in range(self.I)] # concatenate ki hashfunc at lvl i
        self.wi = self.ki / self.ti * np.sqrt(2.0/np.pi) # width of hsfun lvl i
        self.RelVar = lambda mu: np.e**1.854 * 1.0 / np.power(mu, 0.5)
        self.Mi = [int(3* np.ceil( eps**-2 *
                    self.RelVar(self.mui[j]))) for j in range(self.I)]
        print(self.Mi)
        self.HTA = [ELSH(self.Mi[j], X, self.wi[j], self.ki[j]) 
                    for j in range(self.I)]
        
           
    def AMR(self, q):
        """
        Adaptive Mean Relaxation to figure out a constant factor approximation
        to the density KDE(q).
        """
        
        # Mean Relaxation
        for i in range(self.I):
            Z = 0.0
            for j in range(self.Mi[i]):
                Z = Z + self.HTA[i].evalquery(q) 
            print('Iteration {:d}, {:.4f} ? {:.4f}'.format(i,Z / (self.Mi[i]+0.0), self.mui[i] ))
            if Z / (self.Mi[i]+0.0)  >= self.mui[i]:
                print('Success {:d}'.format(i))
                return Z / (self.Mi[i])
        return Z / (self.Mi[i])
        
              
class E2LSH:
    
    def __init__(self, M, X, w, k):
        
        self.HT = [hash_table() for i in range(M)]
        self.scale = w # scale to normalize distances
        self.num = M # number of Hash tables / samples
        self.cur = 0 # current index of Hash Table to query
        self.power = k # concatentation of k hash functions 
        self.N = X.shape[1] #number of points
        for i in range(M):
            self.HT[i].build(X, w, k)
        print('built {:d}-th hash functions'.format(M))   
    
    def prob(self, c):
        """
        returns the collision probability of two points with normalized dist, c
        """
        
        return (erf(1.0 / c) - np.sqrt(2.0 / np.pi)*\
                c * (1 - np.exp(- 1.0 / (2 * c**2))))**self.power
                
    def kernel(self, x,y):
        """
        exponential kernel exp(-||x-y||)
        """
        
        return np.exp(-np.linalg.norm(x-y))
                  
    def evalquery(self, q):
        """
        cycle through hash tables and computes HBE(q)
        """
        
        # find current index
        if self.cur < self.num:
            i = self.cur
        else:
            i = 0
        self.cur = i +1
        
        # Sample a point from the hash bucket
        Xi = self.HT[i].sample(q)
        if not np.array(Xi).any(): 
            return 0.0 # return 0 if empty
        else:
            key = self.HT[i].h(q) # hash query
            c = np.linalg.norm(q - Xi) / self.scale # compute normalized dist
            return self.kernel(Xi, q) / self.prob(c) *\
                        self.HT[i].H[key].num / self.N # return HBE
    
    def RelVar(self, mu):
        """
        Relative Variance Var[Z] / (E[Z])^2 for HBE
        """
        
        return np.e**1.5 * 1.0 / np.sqrt(mu)
           
    def AMR(self, q, acc, fail, lb):
        """
        Adaptive Mean Relaxation to figure out a constant factor approximation
        to the density KDE(q).
        """
        
        eps = 2.0 * acc / 7.0
        c = eps / 2.0   # consistency check parameter
        gamma = eps / 7.0   # rate of relaxation
        delta = 2.0 * acc / np.log(1.0 / lb) / 49.0 * fail # failure prob MoM's
        L = int(np.ceil(9 * np.log(1.0 / delta)))  # Number of means
        Imax = int(np.floor(49.0 / (2.0 * acc) * np.log(1.0/lb)))  # Num rounds
        
        # Initial estimate of mean
        ind = 0
        mui = (1 - gamma) ** ind # guess the value of the mean
        m = int(np.ceil(6 / (eps/3.0)**2 * self.RelVar(mui))) # num to average
        Z = np.zeros(L) # array to store running sums 
        for i in range(L):
            for j in range(m):
                Z[i] = Z[i] + self.evalquery(q)
        # Main loop
        while (abs(np.median(Z/m) - mui) >= c * mui and 
               ind < Imax): # Consistency check 
            ind = ind + 1
            mui = (1 - gamma) * mui # guess
            mnew = int(np.ceil(6.0 / (eps/3.0)**2 * self.RelVar(mui))) #
            for i in range(L):
                for j in range(mnew - m):
                    Z[i] = Z[i] + self.evalquery(q)
            m = mnew
        if ind >= Imax: # If lower bound exceeded
            return 0.0
        else:
            return np.median(Z/m) 
        
        
    def query(self, q, eps, delta, lb):
        """
        Approximates the value of the kernel density mu = KDE(q) >= lb, 
        giving a (1+eps)-approximation with probability at least 1-delta. 
        The first step consits of obtaining a 2-approximation using 
        the Adaptive Mean Relaxation Procedure (AMR).
        """
        
        mu0 = self.AMR(q, 1.0, delta, lb) # 2- approximation to mu
        if mu0 < lb: #if mu0 less than lb return 0
            return 0.0
        else:
            
            L = int(np.ceil(9 * np.log(1.0 / delta))) # median of L means
            m = int(np.ceil(6.0  / eps**2 * self.RelVar(mu0))) # num to average
            Z = np.zeros(L) # array to store running sums/means
            for i in range(L): # median of 
                for j in range(m): # means
                    Z[i] = Z[i] + self.evalquery(q) # running sums
            return np.median(Z/m) # median-of-means
        
    
        
