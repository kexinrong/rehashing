#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:01:09 2018

@author: psimin
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class KDE_Instance:
    
    def __init__(self, kernel,  inv_kernel, num_points, density, 
                 clusters, dimension, scales, spread):
        """
        Creates an instance of a KDE problem using:
            - kernel: a decreasing and invertible function of the distance
            - inv_kernel: the inverse of the kernel function
            - num_points: the nominal number of points per cluster
            - density: the desired kernel density of the origin 0.
            - clusters: number of clusters along random directions
            - dimension: the nominal dimension of the point set
            - scales: number of density scales, between ``density" and "1".
            - spread: what is the relative radius of the cluster with respect
                     to the distance of the cluster center from the origin.
                     
        The function works by picking a certain number of random directions and
        placing clusters of points at certain distances along those random
        directions. 
        """
        
        self.kernel = kernel # kernel to be used
        self.inverse = inv_kernel # inverse of the kernel as a function of dist
        self.num_points = num_points # nominal number of points per direction
        self.num_clusters = clusters # number of random directions/clusters
        self.num_scales = scales # number of distance scales
        
        #Set dimensions
        self.dimension = dimension
        
        # Generate num_clusters many directions
        
        self.D = np.random.randn(self.dimension, self.num_clusters)
        
        # Make the vectors unit norm
        
        for i in range(self.num_clusters):
            self.D[:, i] = self.D[:, i] / np.linalg.norm(self.D[:, i])
            
        # Create relevant distance scales
        
        self.num_scales = int(max([scales, 2])) # always max dist and 0
        dist = self.inverse(density)
        self.scales = np.linspace(0, dist, self.num_scales)
        
        # Calculate number of points in each scale such that the contribution
        # of all distance scale is the same up to constant factors
        
        self.sizes = list(map(lambda x:int(x), np.floor(density*self.num_points 
                         / self.kernel(self.scales))))
        self.N = np.sum(self.sizes) * self.num_clusters
        
        # Generate Points
        
        cnt = 0
        self.X = np.zeros((self.dimension, self.N))
        for i in range(self.num_clusters):
            for j in range(self.num_scales):
                cij = self.D[:,i] * self.scales[j]
                self.X[:,cnt: cnt + self.sizes[j]] = np.outer(cij, 
                 np.ones(self.sizes[j])) + np.random.randn(self.dimension,
                        self.sizes[j]) * spread * self.scales[j] \
                                                 /np.sqrt(self.dimension)
                cnt = cnt + self.sizes[j]                 
    
    
    def merge(self, Data):
        """
        Takes a dimension X m, matrix of data points and augments the instance
        """
        
        S=Data.shape
        assert(S[0]==self.dimension), "Dimensions are not compatible"
        self.X = np.hstack((self.X, Data))
        self.N = self.N + S[1]
        
        
    def query(self, mode="random", dist=0.0):
        """
        Generates query point in R^d according to mode:
            - "random": query is random normal vector with variance dist**2/d
            - "correlated": query is along one of the random directions at dist
        """
        
        if mode == "correlated":
            I = np.random.randint(self.num_clusters)
            return self.D[:, I] * dist
        else:
            return np.random.randn(self.dimension)*dist / \
                    np.sqrt(self.dimension)
#%% Demo
if __name__ == "__main__":
    #%%  Problem Specificaiton
    kernel = lambda x: np.exp(-x)
    inverse = lambda mu: -np.log(mu)
    #%% Creating ``Uncorrelated" instance
    num_points = 100
    clusters = 1000
    scales = 4
    density = 0.01
    dimension = 3
    spread = 0.01
    Instance1 = KDE_Instance(kernel,  inverse, num_points, density, 
                     clusters, dimension, scales, spread)
    
    #%% Plot Dataset
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colors1=[]
    for i in range(Instance1.num_clusters):
        col = np.random.rand(3)
        colors1.extend([col]* (Instance1.N / Instance1.num_clusters))
#        plt.plot([0,Instance.D[0,i]*inverse(density)],
#                 [0,Instance.D[1,i]*inverse(density)],
#                 [0,Instance.D[2,i]*inverse(density)], c=col, ls='--', lw=1, 
#                 alpha=0.2)
    ax = fig.gca(projection='3d')
    ax.scatter(Instance1.X[0,:], Instance1.X[1, :], Instance1.X[2,:],s=10,
               alpha=0.01, c=colors1, edgecolors='face')
    ax.grid(False)
    ax.view_init(25, 35)
    
    #%% Creating Correlated instance
    num_points = Instance1.N
    clusters = 4
    scales = 4
    density = 0.01
    dimension = 3
    spread = 0.01
    Instance = KDE_Instance(kernel,  inverse, num_points, density, 
                     clusters, dimension, scales, spread)
    
    #%% Plot Dataset
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colors=[]
    for i in range(Instance.num_clusters):
        col = np.random.rand(3)
        colors.extend([col]* (Instance.N / Instance.num_clusters))
        plt.plot([0,Instance.D[0,i]*inverse(density)],
                 [0,Instance.D[1,i]*inverse(density)],
                 [0,Instance.D[2,i]*inverse(density)], c=col, ls='--', lw=1, 
                 alpha=0.2)
    ax = fig.gca(projection='3d')
    ax.scatter(Instance.X[0,:], Instance.X[1, :], Instance.X[2,:],s=10,
               alpha=0.01, c=colors, edgecolors='face')
    ax.grid(False)
    ax.view_init(25, 35)    
    
    #%% Creating and Plotting Mixed Instance 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colors=[]
    for i in range(Instance.num_clusters):
        col = np.random.rand(3)
        colors.extend([col]* (Instance.N / Instance.num_clusters))
        plt.plot([0,Instance.D[0,i]*inverse(density)],
                 [0,Instance.D[1,i]*inverse(density)],
                 [0,Instance.D[2,i]*inverse(density)], c=col, ls='--', lw=1, 
                 alpha=0.2)
    Instance.merge(Instance1.X)
    colors.extend(colors1) 
    print(Instance.X.shape[1], len(colors))
    ax.scatter(Instance.X[0,:], Instance.X[1, :], Instance.X[2,:],s=10,
               alpha=0.01, c=colors, edgecolors='face')
    ax.grid(False)
    ax.view_init(25, 35)    