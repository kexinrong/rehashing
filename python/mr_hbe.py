#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 17:23:08 2018

@author: paris
"""
import numpy as np
from scipy.special import erf as erf
import eLSH
import matplotlib.pyplot as plt

class MR_GLSH:
    
    def __init__(self, X, kernel, kernel_inv, T, k_array, w_array, R):
        """
        Input:
        - X is a d x n array of real numbers
        - kernel is a function from non-negative numbers to non-negative number
        - kernel_inv is the inverse function of kernel
        - k_array is a list of T positive integers
        - w_array is a list of T positive real numbers
        Ouput:
        - builds T hashtables where the parameters of each hash function are
          given by k_array, w_array.
        """
        
        assert(T == len(k_array) and T==len(w_array))
        self.HT = [eLSH.hash_table() for i in range(T)]
        self.R = R
        self.T = T
        self.k_array = k_array
        self.w_array = w_array
        self.kernel = kernel
        self.kernel_inv = kernel_inv
        self.N = X.shape[1] # number of points
        for t in range(T):
            self.HT[t].build(X, w_array[t], k_array[t])
            
    def prob(self, dist, t):
        """
        Returns the collision probability for the t-th hash function p_t(dist)
        """
        assert(t in range(self.T))
        c = dist / self.w_array[t] # compute normalized distance
        if c < 0.01:
            return 1.0
        else:
            return (erf(1.0 / c) - np.sqrt(2.0 / np.pi)*\
                c * (1 - np.exp(- 1.0 / (2 * c**2))))**self.k_array[t]
        
    def level_sets(self, mu):
        
        return min([int(np.ceil(np.log(mu * np.exp(2.0 / np.pi / self.R ** 2)) / np.log(0.25))), self.T])
    
    def normalize_const(self, dist, reject_fun, mu, delta):
        """
        Return the sum_{t\in T(r)}p_t^2(r) where r=d(x,y) distance between x, y
        """
        z = 0.0
        for t in range(self.level_sets(mu)):
            # For all t \in T(r) 
            if not reject_fun(self.kernel(dist), self.prob(dist, t),
                              self.prob(self.kernel_inv(mu), t), mu, delta):
                z = z + self.prob(dist, t) ** 2
        return z
    
    def evalquery(self, q, reject_fun, mu, delta):
        """
        cycle through hash tables and computes MR-HBE(q)
        """
        Z = 0.0
        for t in range(self.level_sets(mu)):
            # Sample a point from the hash bucket
            Xt = self.HT[t].sample(q)
            dist = np.linalg.norm(Xt - q)
            weight = self.kernel(dist)
            prob = self.prob(dist, t)
            prob_mu = self.prob(self.kernel_inv(mu), t)
            if not np.array(Xt).any() or \
                    reject_fun(weight, prob, prob_mu, mu, delta):
                pass # do nothing
            else:
                dist = np.linalg.norm(Xt - q)
                key = self.HT[t].h(q) # hash query
                Z = Z + self.kernel(dist) * self.prob(dist, t) /\
                        self.normalize_const(dist, reject_fun, mu, delta) *\
                        self.HT[t].H[key].num / self.N # return HBE 
        return Z
        

class MR_Gauss:
    
    def __init__(self, tau, X, R, delta, eps):
        
        # beta on 1/np.log(1/tau) step
        self.delta = delta
        self.tau = tau
        self.T = int(np.ceil(np.log(1.0/tau)/np.log(4.0)))
        self.mui = [np.exp(- 2.0 / np.pi / R ** 2) * 0.25 ** t for t in range(self.T)]
        beta = [np.sqrt(-np.log(self.mui[t])) / 2.0 for t in range(self.T)]
        self.k_array = [int(np.ceil(np.sqrt(2*np.pi)*beta[t]*R))
                           for t in range(self.T)]
        self.w_array = [self.k_array[t]/beta[t]/np.sqrt(np.pi/2.0) 
                    for t in range(self.T)]
        print(beta)
        print(self.k_array)
        print(self.w_array)
        self.kernel = lambda dist: np.exp(-dist**2)
        self.kernel_inv = lambda mu: np.sqrt(-np.log(mu))
        self.RelVar = lambda mu: 4 *np.exp(3.0/2.0)*mu**(-delta)
        self.M = 500 # 4*int(np.ceil(self.RelVar(tau) / eps**2)) or 500
        print(self.M)
        self.MRA = [MR_GLSH(X, self.kernel, self.kernel_inv, self.T, self.k_array,
                            self.w_array, R) for i in range(self.M)]
        print("Initialization finished")
        self.cur = 0 # current index of Hash Table to query
        
    def next_index(self):
        
        # find current index
        if self.cur < self.M:
            i = self.cur
        else:
            i = 0
        self.cur = i +1
        return i
    
    def reject_fun(self, weight, prob, prob_mu, mu, delta):
        """
        The function is such E[Z^2]\leq \mu^{-delta}
        """
        
        alpha = np.log(weight) / np.log(mu) # w = mu^{alpha}
        beta = np.log(prob)/ np.log(mu) # pt = mu^{beta}
        gamma = np.log(prob_mu) / np.log(mu) # pt_mu = mu^{gamma}
        c =np.log(4*np.exp(3.0/2.0)) / np.log(mu)
        return alpha - 1 + gamma - 2 * beta < - delta + c

    def AMR(self, q):
        """
        Adaptive Mean Relaxation to figure out a constant factor approximation
        to the density KDE(q).
        """
        
        # Mean Relaxation
        for i in range(self.T):
            Z = 0.0
            mi = self.mui[i]
            Mi = int(np.ceil(self.RelVar(mi)))
            for j in range(Mi):
                Z = Z + self.MRA[self.next_index()].evalquery(q, 
                                             self.reject_fun, mi, self.delta)
            if Z / Mi  >= mi:
                print('Success {:d}'.format(i))
                return Z / Mi
        return Z / Mi
    
    def plot_probabilites(self):
        
        dist = np.linspace(0.0001, self.kernel_inv(self.tau), 1000)
        P = np.zeros((self.T, len(dist)))
        for i in range(len(dist)):
            for t in range(self.T):
                weight = self.kernel(dist[i])
                dmu = self.kernel_inv(self.tau)
                p_mu = self.MRA[0].prob(dmu, t)
                pr = self.MRA[0].prob(dist[i], t)
                if self.reject_fun(weight, pr, p_mu, self.tau, self.delta):
                    P[t, i] = 0.0
                else:
                    P[t, i] = self.MRA[0].prob(dist[i], t)
        for t in range(self.T):
            plt.figure()
            plt.plot(dist, P[t,:], c=tuple(np.random.rand(3)))
            plt.title("{:d}-th hash function".format(t+1))
        plt.figure()
        plt.plot(dist, np.max(P[:,:],0))
        plt.title("Max collision probability")

class MR_ELSH:
    
    def __init__(self, X, kernel, kernel_inv, T, k_array, w_array):
        """
        Input:
        - X is a d x n array of real numbers
        - kernel is a function from non-negative numbers to non-negative number
        - kernel_inv is the inverse function of kernel
        - k_array is a list of T positive integers
        - w_array is a list of T positive real numbers
        Ouput:
        - builds T hashtables where the parameters of each hash function are
          given by k_array, w_array.
        """
        
        assert(T == len(k_array) and T==len(w_array))
        self.HT = [eLSH.hash_table() for i in range(T)]
        self.T = T
        self.k_array = k_array
        self.w_array = w_array
        self.kernel = kernel
        self.kernel_inv = kernel_inv
        self.N = X.shape[1] # number of points
        for t in range(T):
            self.HT[t].build(X, w_array[t], k_array[t])
            
    def prob(self, dist, t):
        """
        Returns the collision probability for the t-th hash function p_t(dist)
        """
        assert(t in range(self.T))
        c = dist / self.w_array[t] # compute normalized distance
        if c < 0.01:
            return 1.0
        else:
            return (erf(1.0 / c) - np.sqrt(2.0 / np.pi)*\
                c * (1 - np.exp(- 1.0 / (2 * c**2))))**self.k_array[t]
        
    def level_sets(self, mu):
        
        return min([int(np.ceil(np.log(mu) / np.log(0.25))), self.T])
    
    def normalize_const(self, dist, reject_fun, mu, delta):
        """
        Return the sum_{t\in T(r)}p_t^2(r) where r=d(x,y) distance between x, y
        """
        z = 0.0
        for t in range(self.level_sets(mu)):
            # For all t \in T(r) 
            if not reject_fun(self.kernel(dist), self.prob(dist, t),
                              self.prob(self.kernel_inv(mu), t), mu, delta):
                z = z + self.prob(dist, t) ** 2
        return z
    
    def evalquery(self, q, reject_fun, mu, delta):
        """
        cycle through hash tables and computes MR-HBE(q)
        """
        Z = 0.0
        for t in range(self.level_sets(mu)):
            # Sample a point from the hash bucket
            Xt = self.HT[t].sample(q)
            dist = np.linalg.norm(Xt - q)
            weight = self.kernel(dist)
            prob = self.prob(dist, t)
            prob_mu = self.prob(self.kernel_inv(mu), t)
            if not np.array(Xt).any() or \
                    reject_fun(weight, prob, prob_mu, mu, delta):
                pass # do nothing
            else:
                dist = np.linalg.norm(Xt - q)
                key = self.HT[t].h(q) # hash query
                Z = Z + self.kernel(dist) * self.prob(dist, t) /\
                        self.normalize_const(dist, reject_fun, mu, delta) *\
                        self.HT[t].H[key].num / self.N # return HBE 
        return Z
    
    
class MR_Exp:
    
    def __init__(self, tau, X, R, delta, eps):
        
        # beta on 1/np.log(1/tau) step
        self.delta = delta
        self.tau = tau
        self.eps = eps
        self.T = int(np.ceil(np.log(1.0/tau)/np.log(4.0)))
        self.mui = [0.25 * 0.25 ** t for t in range(self.T)]
        self.k_array = [int(np.sqrt(2*np.pi)*np.ceil(0.5*np.log(1.0/self.mui[t])))
                           for t in range(self.T)]
        self.w_array = [self.k_array[t]/0.5/np.sqrt(np.pi/2.0) 
                    for t in range(self.T)]
        print(self.k_array)
        print(self.w_array)
        self.kernel = lambda dist: np.exp(-dist)
        self.kernel_inv = lambda mu: -np.log(mu)
        self.RelVar = lambda mu: mu**(-delta)
        self.M =  1 #7*int(np.ceil(self.RelVar(tau) / eps**2))
        print(self.M)
        self.MRA = [MR_ELSH(X, self.kernel, self.kernel_inv, self.T, self.k_array,
                            self.w_array) for i in range(self.M)]
        print("Initialization finished")
        self.cur = 0 # current index of Hash Table to query
        
    def next_index(self):
        
        # find current index
        if self.cur < self.M:
            i = self.cur
        else:
            i = 0
        self.cur = i +1
        return i
    
    def reject_fun(self, weight, prob, prob_mu, mu, delta):
        """
        The function is such E[Z^2]\leq \mu^{-delta}
        """
        
        alpha = np.log(weight) / np.log(mu) # w = mu^{alpha}
        beta = np.log(prob)/ np.log(mu) # pt = mu^{beta}
        gamma = np.log(prob_mu) / np.log(mu) # pt_mu = mu^{gamma} 
        return alpha - 1 + gamma - 2 * beta < - delta  

    def AMR(self, q):
        """
        Adaptive Mean Relaxation to figure out a constant factor approximation
        to the density KDE(q).
        """
        
        # Mean Relaxation
        for i in range(self.T):
            Z = 0.0
            mi = self.mui[i]
            Mi = 7*int(np.ceil(self.RelVar(mi)/self.eps**2))
            for j in range(Mi):
                Z = Z + self.MRA[self.next_index()].evalquery(q, 
                                             self.reject_fun, mi, self.delta)
            if Z / Mi  >= mi:
                print('Success {:d}'.format(i))
                return Z / Mi
        return Z / Mi
    
    def plot_probabilites(self):
        
        dist = np.linspace(0.0001, self.kernel_inv(self.tau), 1000)
        P = np.zeros((self.T, len(dist)))
        for i in range(len(dist)):
            for t in range(self.T):
                weight = self.kernel(dist[i])
                dmu = self.kernel_inv(self.tau)
                p_mu = self.MRA[0].prob(dmu, t)
                pr = self.MRA[0].prob(dist[i], t)
                if self.reject_fun(weight, pr, p_mu, self.tau, self.delta):
                    P[t, i] = 0.0
                else:
                    P[t, i] = self.MRA[0].prob(dist[i], t)
        for t in range(self.T):
            plt.figure()
            plt.plot(dist, P[t,:], c=tuple(np.random.rand(3)))
            plt.title("{:d}-th hash function".format(t+1))
        plt.figure()
        plt.plot(dist, np.max(P[:,:],0))
        plt.title("Max collision probability")
        