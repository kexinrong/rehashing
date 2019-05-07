#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Paris Siminelakis
@email: psiminelakis@gmail.com
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

    def __init__(self, M, X, w, k, kernel_fun= lambda x, y :\
                    np.exp(-np.linalg.norm(x-y))):

        self.HT = [hash_table() for i in range(M)]
        self.scale = w # scale to normalize distances
        self.num = M # number of Hash tables / samples
        self.cur = 0 # current index of Hash Table to query
        self.power = k # concatentation of k hash functions
        self.N = X.shape[1] #number of points
        self.kernel = kernel_fun
        for i in range(M):
            self.HT[i].build(X, w, k)
        print('built {} hash functions'.format(M))

    def prob(self, c):
        """
        returns the collision probability of two points with normalized dist, c
        """

        return (erf(1.0 / c) - np.sqrt(2.0 / np.pi)*\
                c * (1 - np.exp(- 1.0 / (2 * c**2))))**self.power


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

class GHBE:

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
        self.wi = self.ki / np.maximum(self.ti, 1) * np.sqrt(2.0/np.pi)
        self.RelVar = lambda mu: np.e**1.854 * 1.0 / np.power(mu, 0.75)
        self.Mi = [int(3* np.ceil( eps**-2 *
                    self.RelVar(self.mui[j]))) for j in range(self.I)]
        print(self.Mi)
        self.HTA = [ELSH(self.Mi[j], X, self.wi[j], self.ki[j],\
                         lambda x, y : np.exp(-np.linalg.norm(x-y)**2))\
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

class EHBE:

    def __init__(self, X, tau, eps):
        self. eps = eps
        self.R = np.log(1 / tau) # effective diameter of the set
        self.gamma = 0.5           # rate that we decrease our estimate
        self.I = int(np.ceil(np.log2(1 / tau))) # number of different guesses
        self.mui = np.array([(1 - self.gamma) ** i for i in range(self.I)])
        self.k = int(np.ceil(np.sqrt(2*np.pi)*self.R*np.log(1/tau)))
        self.w = np.sqrt(2 / np.pi) * 2 * self.k
        self.RelVar = lambda mu: np.e**1.854 * 1.0 / np.power(mu, 0.5)
        self.Mi = [int(3* np.ceil( eps**-2 *
                    self.RelVar(self.mui[j]))) for j in range(self.I)]
        self.HTA = [ELSH(self.Mi[j], X, self.w, self.k)
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

#%% Demo
if __name__ == "__main__":
    import KDE_instance as kde
    #%%  Problem Specificaiton for Gaussian Kernel
    kernel = lambda x: np.exp(-x**2)
    inverse = lambda mu: np.sqrt(- np.log(mu))
    #%% Creating ``Uncorrelated" instance
    num_points = 30
    clusters = 60
    scales = 3
    density = 0.01
    dimension = 30
    spread = 0.02
    Instance1 = kde.KDE_instance(kernel,  inverse, num_points, density,
                     clusters, dimension, scales, spread)
    n = Instance1.N
    #%%
    eps = 0.5
    tau = float(10**-3)
    kde1 = GHBE(Instance1.X, tau, eps)

    #%% Unit Test for (n,k,d,s)-Instance
    print "================================"
    print "Unit Test for Gaussian Kernel and random queries"
    print "================================"
    print "Required relative error < {:.3f}".format(eps)
    print "--------------------------------"

    iterations = 20
    cnt = 0
    for j in range(iterations):
        # Random queries around 0
        q = np.zeros(Instance1.dimension) + np.random.randn(dimension) / np.sqrt(dimension)
        kernel_fun = lambda x,y: np.exp(-np.linalg.norm(x-y)**2)

        kd = 0.0
        for i in range(n):
            kd = kd + kernel_fun(q, Instance1.X[:,i])
        kd = kd / n
        est = kde1.AMR(q)
        print ("Estimate: {:f} True: {:f}".format(est, kd))
        if abs((kd - est) / kd) <= eps:
            cnt = cnt + 1
        print "Query {} rel-error: {:.3f}".format(j+1,(kd - est) / kd)

    print "--------------------------------"
    print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
    print "================================"

    #%%  Problem Specificaiton for Exponential Kernel
    kernel = lambda x: np.exp(-x)
    inverse = lambda mu: - np.log(mu)
    #%% Creating ``Uncorrelated" instance
    num_points = 30
    clusters = 60
    scales = 3
    density = 0.01
    dimension = 30
    spread = 0.02
    Instance2 = kde.KDE_instance(kernel,  inverse, num_points, density,
                     clusters, dimension, scales, spread)
    n = Instance2.N
    #%%
    eps = 0.5
    tau = float(10**-3)
    kde2 = EHBE(Instance2.X, tau, eps)

    #%% Unit Test for (n,k,d,s)-Instance
    print "================================"
    print "Unit Test for Exponential kernel and random queries"
    print "================================"
    print "Required relative error < {:.3f}".format(eps)
    print "--------------------------------"

    iterations = 20
    cnt = 0
    for j in range(iterations):
        # Random queries around 0
        q = np.zeros(Instance2.dimension) + np.random.randn(dimension) / np.sqrt(dimension)
        kernel_fun = lambda x,y: np.exp(-np.linalg.norm(x-y))

        kd = 0.0
        for i in range(n):
            kd = kd + kernel_fun(q, Instance2.X[:,i])
        kd = kd / n
        est = kde2.AMR(q)
        print ("Estimate: {:f} True: {:f}".format(est, kd))
        if abs((kd - est) / kd) <= eps:
            cnt = cnt + 1
        print "Query {} rel-error: {:.3f}".format(j+1,(kd - est) / kd)

    print "--------------------------------"
    print "Failure prob: {:.2f}".format(1 - cnt / float(iterations))
    print "================================"
