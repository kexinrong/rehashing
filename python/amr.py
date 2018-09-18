#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:48:37 2017

@author: paris
@email: psiminelakis@gmail.com

This is an implementation of the Adaptive Mean Relaxation algorithm for
estimating the mean of a random V-bounded random variable.

If you use this package you should cite: 
    
    "Hashing-Based-Estimators for Kernel Density in High Dimensions"
    Moses Charikar, Paris Siminelakis, FOCS 2017.
"""

import types as tp
from math import ceil, log, floor as ceil, log, floor
import numpy as np

class SamplingOracle:
    
    def __init__(self, oracle, imin, imax):
        """
        SamplingOracle(oracle, imin, imax): returns a sampling oracle that ret-
                                            urns samples by calling the next 
                                            method.
        """
        # Typechecking
        assert type(imin) is tp.IntType, \
            "Variable imin is not an integer %r" % imin
        assert type(imax) is tp.IntType, \
            "Variable imax is not an integer %r" % imax
        assert isinstance(oracle, tp.FunctionType), \
            "Variable oracle is not a function %r" % oracle
        # Logic checking
        assert imin <= imax, \
            "Variable imin is larger than imax"
        
        # Initialize
        self.soracle = oracle
        self.imin = imin
        self.imax = imax
        self.icur = imin
        
    def next(self):
        """
        SamplingOracle.next(): returns a fresh sample from the oracle as long
                                as such a sample is available
        """
        if self.icur <= self.imax:
            self.icur = self.icur + 1        
            return self.soracle(self.icur - 1)
        else:
            self.icur = 1 
            return self.soracle(self.icur - 1)
        
    def mean_relaxation(self, acc, fail, lb, RelVar):
        """
        MeanRelaxation: adaptively estimates the mean $\mu$ of a random variable
                        given access to an oracle returning samples. For the algor-
                        ithm to work $RelVar(\mu) * \mu^2$ needs to be inreasing
                        with $\mu$ and $RelVar(\mu)$ needs to be decreasing.
        
        Input: - 
            
        """
        
        # Typechecking
        assert type(acc) is tp.FloatType, \
            "Variable acc is not a float %r" % acc
        assert type(fail) is tp.FloatType, \
            "Variable fail is not a float %r" % fail
        assert type(lb) is tp.FloatType, \
            "Variable lb is not a float %r" % lb
        assert isinstance(RelVar, tp.FunctionType), \
            "Variable RelVar is not a function %r" % RelVar
        # Logic checking
        assert acc <= 1 and acc > 0, \
            "Variable acc is not in (0,1]: %r" % acc
        assert fail < 1 and fail > 0, \
            "Variable fail is not in (0,1): %r" % fail
        assert lb <1 and lb > 0, \
            "Variable lb is not in (0,1): %r" % lb
        
        eps = 2.0 * acc / 7.0
        c = eps / 2.0   # consistency check parameter
        gamma = eps / 7.0   # rate of relaxation
        delta = 2.0 * acc / log(1.0 / lb) / 49.0 * fail # failure prob MoM's
        L = int(ceil(9 * log(1.0 / delta)))  # Number of means
        Imax = int(floor(49.0 / (2.0 * acc) * log(1.0/lb)))  # Number of rounds
        
        # Initial estimate of mean
        ind = 0
        mui = (1 - gamma) ** ind # guess the value of the mean
        m = int(ceil(6  / (eps/3.0)**2 * RelVar(mui))) # num of samples to average
        Z = np.zeros(L) # array to store running sums 
        for i in range(L):
            for j in range(m):
                Z[i] = Z[i] + self.next()
        # Main loop
        while (abs(np.median(Z/m) - mui) >= c * mui and 
               ind < Imax): # Consistency check 
            ind = ind + 1
            mui = (1 - gamma) * mui # guess
            mnew = int(ceil(6  / (eps/3.0)**2 * RelVar(mui))) # num to average
            for i in range(L):
                for j in range(mnew - m):
                    Z[i] = Z[i] + self.next()
            m = mnew
        if ind >= Imax: # If lower bound exceeded
            return np.median(Z/m) # or 0.0
        else:
            return np.median(Z/m) 
