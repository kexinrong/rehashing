#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Paris Siminelakis
@email: psiminelakis@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Ellipse

class KDE_instance:

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
    num_points = 30
    clusters = 60
    scales = 4
    density = 0.01
    dimension = 2
    spread = 0.02
    Instance1 = KDE_instance(kernel,  inverse, num_points, density,
                     clusters, dimension, scales, spread)

    #%% Plot Dataset
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    colors1=[]
    ells = [Ellipse(xy=np.zeros(2),
                    width=2*Instance1.scales[j], height=2*Instance1.scales[j],
                    angle=0.0, edgecolor='k')
            for j in range(Instance1.num_scales)]
    cel=['r','m','b','k']
    for j in range(Instance1.num_scales):
        ax.add_artist(ells[j])
        ells[j].set_clip_box(ax.bbox)
        ells[j].set_alpha(0.225)
        ells[j].set_facecolor('k')
    ax.text(-0.75 , 0, '$D   \leq$', fontsize=11)
    for j in range(Instance1.num_scales):
        ax.text(Instance1.scales[j] +  Instance1.scales[1] - 1.2 , 0, \
                '$D_{'+str(j+1)+'}$', fontsize=11, color=cel[j])
    for i in range(Instance1.num_clusters):
        col = np.random.rand(3)
        colors1.extend([col]* (Instance1.N / Instance1.num_clusters))
    ax.scatter(Instance1.X[0,:], Instance1.X[1, :],s=10,
               alpha=0.9, c=colors1, edgecolors='face')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(False)
    plt.savefig('clusters.png')
    plt.show()

    #%% Creating Correlated instance
    num_points = Instance1.N
    clusters = 3
    scales = 4
    density = 0.01
    dimension = 2
    spread = 0.05
    Instance = KDE_instance(kernel,  inverse, num_points, density,
                     clusters, dimension, scales, spread)

    #%% Plot Dataset
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    colors=[]
    for i in range(Instance.num_clusters):
        col = np.random.rand(3)
        colors.extend([col]* (Instance.N / Instance.num_clusters))
        plt.plot([0,Instance.D[0,i]*inverse(density)],
                 [0,Instance.D[1,i]*inverse(density)], c=col, ls='--', lw=1,
                 alpha=0.6)
    ax.scatter(Instance.X[0,:], Instance.X[1, :], s=10,
               alpha=0.6, c=colors, edgecolors='face')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(False)
    plt.savefig('correlated.png')
    plt.show()

    #%% Creating and Plotting Mixed Instance
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for i in range(Instance.num_clusters):
        plt.plot([0,Instance.D[0,i]*inverse(density)],
                 [0,Instance.D[1,i]*inverse(density)], c='k', ls='--', lw=1,
                 alpha=0.6)
    ax.scatter(Instance.X[0,:], Instance.X[1, :],s=10,
               alpha=0.6, c=colors, edgecolors='face')
    ax.scatter(Instance1.X[0,:], Instance1.X[1, :],s=10,
                   alpha=0.9, c=colors1, edgecolors='face')

    ax.grid(False)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.savefig('generic.png')
    plt.show()
