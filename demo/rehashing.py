#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Paris Siminelakis
@email: psiminelakis@gmail.com
"""

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.special import erf as erf
from functools import partial as partial

MIN_DIST = 10**-6 # threshold for considering two points distinct
CIRCLE_POINTS = 10**3 # number of points used to visualize circles.

def make_circle(r, m):
    """
        make_circle: returns a 2 X m matrix of m points distributed on a
                     circle around (0,0)

        Input:
            - r > 0: radius of the circle
            - m : number of points on the circle

        Output:
            - 2 X m numpy array

        Example: circle_points = make_circle(1, 1000)
    """

    t = np.linspace(0, np.pi * 2.0, m)
    t = t.reshape((m, 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))


def drawAnnuli(r, R, m, ax, transparency):
    """
        drawAnnuli: draws a semi-transparent annulus around 0 with radii r < R

        Input:
            - r >= 0 : inner radius of the annulus
            - R >= r : outer radius of the annulus
            - ax: axis handle (matplotlib.pyplot) where the annulus is drawn
            - transparency in [0,1]: alpha value used to plot annulus body
    """

    inside_vertices = make_circle(r, m) # inner circle
    outside_vertices = make_circle(R, m) # outer circle

    codes = np.ones(m, dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO

    vertices = np.concatenate((outside_vertices[::1],
                               inside_vertices[::-1]))
    all_codes = np.concatenate((codes, codes))
    path = mpath.Path(vertices, all_codes)
    patch = mpatches.PathPatch(path,alpha=transparency, facecolor='b', \
                                edgecolor='none')
    ax.add_patch(patch)


def eLSH_prob(x, y, w, k):
    """
        eLSH_prob: collision probability of the hashing scheme of Datar,
                   Immorlica, Indyk, Motwani SoCG 2004.
        Input:
            - x:  array/vector
            - y: array/vector
            - w > 0: width of the hash buckets
            - k >= 0: number of i.i.d hash function to concatenate (power)

        Output:
            - P[h(x)=h(y)] ^ k : where h(x) = ceil(<g, x> / w + b)
                                 with g ~ N(0,I_d) append and b~U[0,1]
    """

    dist = np.linalg.norm(x - y) # euclidean distance between two points
    c = dist / w # normalize by bucket width
    if c > MIN_DIST: # if points are not almost identical
        return (erf(1.0 / c) - np.sqrt(2.0 / np.pi)*\
                c * (1 - np.exp(- 1.0 / (2 * c**2)))) ** k
    return 1.0 # if points are almost identical

def weighted_sampling(points, weights, size):
    """
        weighted_sampling: return a set of i.i.d. points from where point i is
                           sampled with probability proportional to weight i

        Input:
            - points: d x n array of n points
            - weights: vector of n non-negative numbers
            - size: number of random points to return

        Output:
            - sub_sample: a dictionary such that
                -- sub_sample['p']: d x size array
                -- sub_sample['w']: array of non-negative weights that sum to 1
    """

    (d,n) = points.shape

    # sampling
    indices = np.random.choice(np.arange(n), size=size, \
                                p=weights / np.sum(weights))
    sub_sample = {}
    sub_sample['p'] = points[:, indices]
    sub_sample['w'] = np.ones(len(indices)) / len(indices)

    return sub_sample

def kcenter(Data, k, seed=0):
    """
        kcenter: finds k points (centers) to approximately minimize the maximum
                 distance

        Input:
            - Data: d x n array of n points
            - k: number of centers to select
            - seed: index of the first center to select (default = 0)

        Output: (centers, max_dist)
            - centers: a list of k indices of the centers
            - max_dist: the maximum distance from any point to its
                        closest center

        Example: (centers_ids, max_dist) = kcenter(Data, 10, 0)
    """

    n = Data.shape[1]
    centers = [seed] # start with an arbitrary center
    for i in range(k-1):
        max_idx = 0
        max_dist = 0
        for j in range(n):
            min_dist = np.iinfo(np.int32).max #initialize min_dist = infinity
            for c in centers:
                if np.linalg.norm(Data[:,c] - Data[:, j]) <= min_dist:
                    min_dist = np.linalg.norm(Data[:,c] - Data[:, j])
            if min_dist > max_dist:
                max_dist = min_dist
                max_idx = j
        centers.append(max_idx)
    return (centers, max_dist)


def set_parameters_kcenter(k, n):
    """
        set_parameters_kcenter: sets parameters for k-center so that
                                    k ^ 3 + k * n = O(n)

        Input:
            - k: the target number of centers/size of the sketch
            - n: the size of the data set to sketch

        Output: (k, n_feasible)
            - k: the feasible number of centers
            - n_feasible: the size of the sub-sample upon which to apply
                          k-center so that k * n_feasible = O(n)

        Example: (k_feasible, n_feasible) = set_parameters_kcenter(k, n)
    """

    if k**3 > n * k:
        k = int(n**(1.0/3))
    n_feasible = min([int(np.floor(n / k)), n])
    return (k, n_feasible)


class rehashing:

    def __init__(self, points=[], weights=[],\
                 kernel_fun=lambda x,y: np.exp(-np.linalg(x - y)**2)):
        """
            rehashing: initialize class instance by specifying points, weights,
                       and kernel function

            Input:
                - points: d x n numpy array of points
                - weights: n numpy array of non-negative weights
                - kernel_fun: function handle that takes two points as inputs
                              and outputs a non-negative number

            Example: instance = rehashing(np.random.randn(10, 1000),
                                          np.random.rand(1000),
                                          lambda x,y: np.exp(-np.linalg(x-y)))
        """

        self.points = points
        self.weights = weights
        self.num_of_points = len(weights)
        self.kernel = kernel_fun


    def eval_density(self, query_point):
        """
            eval_density: computes the exact kernel density at a query point on
                          the whole dataset

            Input:
                - query_point: a numpy array/vector

            Output:
                - kernel density sum_i { w_i * k(x_i, query_point) }

            Example: instance.eval_density(query_point)
        """

        z = 0
        for i in range(self.num_of_points):
                z = z + self.kernel(self.points[i], query_point)\
                                * self.weights[i]
        return z


    def create_sketch(self, method='ska', accuracy=0.2, threshold=10**-4,
               num_of_means=3):
        """
            create_sketch: creates sketches to approximate the kernel density

            Input:
                - method in {"random", "ska"}:
                    -- "random": creates a sketch by sampling points according
                                to self.weights
                    -- "ska": creates a sketch using a combination of
                              Sparse Kernel Approximation (Cortes and Scott'15)
                              and random sample so that the pre-processing time
                              is O(n).
                - accuracy in (0,1): the multiplicative accuracy of approximati-
                                     on of the sketch
                - threshold in (0,1): the minimal density that we would like
                                      to approximate up to a multiplicative
                                      accuracy
                - num_of_means: specifies the number of sketches (means) of
                                which the median is the final estimate

            Example: instance.create_sketch('ska', 0.2, 10**-4, 3)
        """

        self.num_of_means = num_of_means
        self.size_of_sketch = int(6.0 / accuracy ** 2 / threshold)
        self.sketches = []

        if method == 'random':
            # Creates random sketches of the data
            for l in range(self.num_of_means):
                sketch_l = weighted_sampling(self.points, \
                                             self.weights, \
                                             self.size_of_sketch)
                self.sketches.append(sketch_l)

        if method == 'ska':
            # Implements the approach of Cortes and Scott 2015 using k-center
            # under the constraint that the total pre-perocessing time is
            # linear in thenumber of points.
            for l in range(self.num_of_means):
                # find valid parameters for k-center that satisfy constraints
                (k, n_feasible) = set_parameters_kcenter(self.size_of_sketch, n)
                # sub-sample data set so that there n_feasible points
                sub_sample = weighted_sampling(self.points, \
                                               self.weights,\
                                               n_feasible)
                # run kcenter to obtain the points
                centers, max_dist = kcenter(sub_sample['p'], k, 0)
                # compute center densities
                y = np.zeros(k)
                for i in range(k):
                    for j in range(self.num_of_points):
                        y[i] = y[i] + self.kernel(self.points[:, j],\
                                sub_sample['p'][:,centers[i]]) / \
                                self.num_of_points
                # kernel matrix between centers
                K = np.array([[self.kernel(sub_sample['p'][:,centers[ci]], \
                             sub_sample['p'][:,centers[cj]]) \
                             for cj in centers]  for ci in centers])
                # Least squares fit of center densities
                w  = np.linalg.pinv(K).dot(y)
                # complement the sketch with random samples
                num_of_random = self.size_of_sketch - k
                random_sketch = weighted_sampling(self.points,
                                                  self.weights,\
                                                  num_of_random)
                # merge the two sketches by weighting them appropriately.
                sketch_l = {}
                sketch_l['p'] = np.append(sub_sample['p'][:, centers], \
                                            random_sketch['p'])
                sketch_l['w'] = np.append(w / k, (1.0 - 1.0 /k) * \
                                          random_sketch['w'])

                self.sketches.append(sketch_l)


    def eval_sketch(self, query_point):
        """
            eval_sketch: evaluates the sketches created by create_sketch at a
                         a query point

            Input:
                - query_point: numpy_array/vector specifying query point

            Ouput:
                - estimate of the density at that query point

            Example: instance.eval_sketch(query_point)
        """

        z = np.zeros(self.num_of_means)
        for l in range(self.num_of_means):
            for i in range(self.size_of_sketch):
                z[l] = z[l] + self.kernel(self.sketches[l]['p'][:, i], \
                                          query_point) \
                                * self.sketches[l]['w'][i]
        return np.median(z)


    def AMR_random(self, q, acc=0.2, threshold=10**-4):
        """
            AMR_random: implements Adaptive Mean Relaxation to figure out
                        a constant factor approximation to the density using
                        random sampling as the underlying unbiased estimator

            Input:
                - q: numpy array that specifies the query point
                - acc in (0,1): multiplicative accuracy to estimate density
                - threshold in (0,1): lower density that we wish to approximate

            Output: (Zmed, rand_ind)
                - Zmed: estimate of the density
                - rand_ind: the indices of points used to approximate
                            the density

            Example: (Z, random_indices) = instance.AMR_random(q, 0.2, 10**-4)
        """

        # Mean Relaxation
        I = int(np.ceil(np.log2(1 / (acc * threshold)))) # max number of levels
        for i in range(I):
            Z = np.zeros(self.num_of_means)
            mu_i = 2**-(i+1) # guess of the density
            Mi = int(6 / acc**2 / mu_i * 2) # number of samples required
            for l in range(self.num_of_means):
                rand_ind = np.random.choice(np.arange(self.size_of_sketch),
                                                  size=Mi, \
                                  p=self.sketches[l]['w']/\
                                  np.sum(self.sketches[l]['w']))
                for j in range(Mi): # Estimate the density
                    Z[l] = Z[l] + self.kernel((self.sketches[l]['p'])[:,\
                                                rand_ind[j]], q) / Mi
            Zmed = np.median(Z)
            if Zmed  >= mu_i: # If guess is a lower bound on the estimate return
                print('Success {:d}: {:f}'.format(i, Zmed))
                return (Zmed, rand_ind)
        return (Zmed, rand_ind)

    def compute_lambdas(self, query_point, data_points, eps, mu_0):
        """
            compute_lambdas: estimate two thresholds lambda <= Lambda such that
                             the contribution of weights between these two
                             thresholds is at least an (1-eps) fraction of the
                             density at a query point

            Input:
                - query_point: numpy array that specifies query point
                - data_points: d x m array that specifies m sample points
                - eps in (0,1): used to set the fraction of the data_points
                                that have weight within the two thresholds
                - mu_0: density at the query point

            Ouput: (lambda_eps, Lambda_eps)
                - where lambda_eps <= Lambda_eps

            Example: instance.compute_lambdas(query_point, data, 0.1, q_density)
        """

        n = data_points.shape[1]
        w = np.zeros(n)
        # comute weights of data_points by evaluating the kernel function
        for i in range(n):
            w[i] = self.kernel(query_point, data_points[:, i])
        w.sort() # sort in increasing order
        # find the first index with weight at least mu_0
        id0 = np.argmax(w >= mu_0)
        # compute the contribution of points with weight less than mu_0
        if id0 > 0:
            mu_4 = np.sum(w[:id0]) / n
        else:
            mu_4 = 0.0
        w0 = w[id0:] # vector wuth weights at least mu_0
        wc1 = np.cumsum(w0 / n)
        # find the last index such that the contribution of points is less than
        # epsilon fration of the query density
        id1 = np.argmax(wc1 >= (eps * mu_0 - mu_4) / 2)
        lambda_eps = w0[id1]
        wc2 = np.cumsum(np.flipud(w0 / n))
        # find the first index such that the contribution of points is less than
        # epsilon fration of the query density
        id2 = np.argmax(wc2 >= (eps * mu_0 - mu_4) / 2)
        Lambda_eps = np.flipud(w0)[id2]

        return (lambda_eps, Lambda_eps)


    def variance_bound(self, query_point,  data_points, collision_prob, \
                       lambda_eps, Lambda_eps, mu_0):
        """
            variance_bound: using a set of points and a query, estimates the
                            variance of Random Sampling and HBE
                            schemes with certain collision probabilities
                            according to Lemma 4 in Siminelakis, Rong, Bailis,
                            Charikar, Levis ICML 2019.

            Input:
                - query_point: numpy array specifiying the query point
                - data_points: d x m numpy array of points
                - collision_prob: list of function handles, where each element
                                  of the list corresponds to a collision
                                  probability p(x,y) of a hash function
                - lambda_eps, Lambda_eps: thresholds for kernel values used
                                            to partitions points into sets
                - mu_0: estimate of the query densitylinear in the

            Output: (V_RS, V_H)
                - V_RS: is the estimate of the variance of random sampling
                - V_H: is an array where the t-th element corresponds to the
                        estimate of the variance for the t-th hashing scheme.

            Example:
                - (V_RS,V_H) = instance.variance_bound(query_point,S0_points,\
                                                collision_prob, lambda_eps, \
                                                Lambda_eps, mu_0)
        """

        n = data_points.shape[1]
        w = np.zeros(n) # initialize the vector of weights (kernel values)
        for i in range(n):
            w[i] = self.kernel(query_point, data_points[:, i])
        w_sorted = np.sort(w) # sort in increasing order
        ind_sorted = np.argsort(w)
        ind4 = np.argmax(w_sorted >= mu_0)
        # set S_4 contains indices 0:ind4
        mu4 = np.sum(w_sorted[:ind4]) / n
        ind3 = np.argmax(w_sorted >= lambda_eps)
        # set S_3 contains indices ind4:ind3
        mu3 = np.sum(w_sorted[ind4:ind3]) / n
        ind2 = np.argmax(w_sorted >= Lambda_eps)
        # set S_2contains indices ind3:ind2
        mu2 = np.sum(w_sorted[ind3:ind2]) / n
        # set S_1 contains indices ind2:end
        mu1 = np.sum(w_sorted[ind2:]) / n
        # Computes variance bound of Lemma 4 for Random Sampling Vij = 1
        V_RS = w_sorted[-1] / w_sorted[ind2] * mu1**2 +\
                + w_sorted[-1] / w_sorted[ind3] * mu1 * mu2 +\
                + w_sorted[-1] / w_sorted[ind4] * mu1 * mu3 +\
                + w_sorted[-1] * mu1 * ind4 / n +\
                + w_sorted[ind2] / w_sorted[ind3] * mu2**2 +\
                + w_sorted[ind2] / w_sorted[ind4] * mu2 * mu3 +\
                + w_sorted[ind2] * mu2 * ind4 / n\
                + w_sorted[ind3] / w_sorted[ind4] * mu3**2 +\
                + w_sorted[ind3] * mu3 * ind4 / n +\
                + w_sorted[ind4] * mu4
        # Initialize array to store variance estimates for hashing schemes
        V_H = np.zeros(len(collision_prob))
        # estimates variance of t-th hashing scheme
        for t in range(len(collision_prob)):
            p = np.zeros(n) # initialize the vector of probabilities
            for i in range(n):
                p[i] = collision_prob[t](query_point, data_points[:, i])
            # sort according to the order the weights were sorted
            # We assume beyond this point that this is a valid ordering of
            # the probabilities as well, i.e., they are increasing function of
            # the weights
            p_sorted = p[ind_sorted]
            # Computes variance bound of Lemma 4 for an HBE with
            # Vij = min{p_i, p_j} / p_i^2
            V_H[t] = w_sorted[-1] / w_sorted[ind2] * p_sorted[ind2] / \
                        p_sorted[-1]**2 * mu1**2 +\
                        + w_sorted[-1] / w_sorted[ind3] * p_sorted[ind3] /\
                           p_sorted[-1]**2 * mu1 * mu2 +\
                        + w_sorted[-1] / w_sorted[ind4] * p_sorted[ind4] /\
                           p_sorted[-1]**2 * mu1 * mu3 +\
                        + w_sorted[-1] * p_sorted[ind4] /\
                           p_sorted[-1]**2 * mu1 * ind4 / n + \
                        + w_sorted[ind2] / w_sorted[ind3] * p_sorted[ind3] /\
                           p_sorted[ind2]**2 * mu2**2 +\
                        + w_sorted[ind2] / w_sorted[ind4] * p_sorted[ind4] /\
                           p_sorted[ind2]**2 * mu2 * mu3 +\
                        + w_sorted[ind2] * mu2 * p_sorted[ind4] /\
                           p_sorted[ind2]**2 * ind4 / n \
                        + w_sorted[ind3] / w_sorted[ind4] * p_sorted[ind4] /\
                           p_sorted[ind3]**2 * mu3**2 +\
                        + w_sorted[ind3] * p_sorted[ind4] /\
                           p_sorted[ind3]**2 * mu3 * ind4 / n +\
                        + w_sorted[ind4] * p_sorted[ind4] /\
                           p_sorted[ind4]**2 * mu4
        return (V_RS, V_H)


    def log_condition_plot(self, lambda_eps, Lambda_eps, threshold=10**-4):
        """
            log_condition_plot: uses a sequence of lambda_eps and Lambda_eps to
                                produce a logCondition plot according to
                                Siminelakis, Rong, Bailis, Charikar, Levis ICML
                                2019 and visualize the local query structure of
                                the corresponding dataset that produced these
                                sequences.

            Input:
                - lambda_eps: numpy array with elements in (0,1]
                - Lambda_eps: numpy array with elements in (0,1] such that
                                Lambda_eps >= lambda_eps
                - threshold: lower density of interest.

            Output:
                - plots overlapping spherical annuli around (0,0) to visualize
                  the query structure of the dataset.

            Example: instance.log_condition_plot(lambda_seq, Lambda_seq, 10**-4)
        """

        T = len(lambda_eps)
        # set transparency to indicate the fraction of the queries that have
        # non-trivial contribution from points with certain weights
        alpha = 1.0 / T
        # create sequence of outer and inner radii for the annuli
        R_seq = - np.log(lambda_eps) # outer radii
        r_seq = - np.log(Lambda_eps) # inner radii

        #Path = mpath.Path
        fig, ax = plt.subplots()

        for t in range(T):
            drawAnnuli(r_seq[t], R_seq[t], CIRCLE_POINTS, ax, alpha)

        ax.set_xlim(np.log(threshold), -np.log(threshold))
        ax.set_ylim(np.log(threshold), -np.log(threshold))
        ax.set_title('Dataset Visualization')
        ax.set_aspect(1.0)
        plt.show()


    def diagnostic(self, collision_prob, acc=0.2, threshold=10**-4, \
                   num_queries=100, visualization_flag=True):
        """
            diagnostic: procedure that estimates the relative variance of
                        Random Sampling and HBE's with certain collision
                        probabilities and plots the local query structure of
                        the dataset according to Siminelakis, Rong, Bailis,
                        Charikar, Levis ICML 2019.

            Input:
                - collision_prob: a list of function handles where each element
                                  corresponds to a collision probability p(x,y)
                                  of a hashing scheme
                - acc in (0,1): desired multiplicative approximation
                - threshold: lower density we wish to be able to approximate
                - num_queries: number of queries used to estimate relative
                               variance and visualize query structure
                - visualization_flag: if True also creates log-condition plot

            Output: (rV_RS_mean, rV_H_mean)
                - rV_RS_mean: estimate of the mean number of random samples
                              required to estimate the query up to a constant
                - rV_H_mean: estimate of the mean number of hash_tables created
                             with hash functions with certain collision probabi-
                             lities that are required to estimate query density.

            Example: instance.diagnostic(collision_prob, 0.2, 10**-4, 100, True)
        """

        print("Diagnostic procedure:")
        # initalize sequence of lambdas
        lambda_eps = np.zeros(num_queries)
        Lambda_eps = np.zeros(num_queries)
        # initalize estimates for Relative Variance
        rV_RS = np.zeros(num_queries)
        rV_H = np.zeros((len(collision_prob), num_queries))

        for t in range(num_queries):
            print('Query {:d}'.format(t+1))
            # generate a random query from the dataset
            query_point = self.points[:, np.random.randint(self.num_of_points)]
            # use the adaptive procedure to estimate the density and get a
            # representative set of points S0.
            (mu_0, S0) = self.AMR_random(query_point, acc, threshold)
            S0_points = self.sketches[0]['p'][:, S0]
            # use the representative set of points and density to compute
            # two thresholds lambda_eps <= Lambda_eps
            (lambda_eps[t], Lambda_eps[t]) = self.compute_lambdas(query_point, \
                                                         S0_points, eps, mu_0)
            print('lambda = {:f}, Lambda = {:f}'.format(lambda_eps[t], \
                                                        Lambda_eps[t]))
            # use the variance bound to estimate the variance of the estimators
            (V_RS, V_H) = self.variance_bound(query_point,  S0_points, \
                                              collision_prob, lambda_eps[t], \
                                              Lambda_eps[t], mu_0)
            # Estimate the relative variance of the estimators for queries of
            # density >= threshold
            rV_RS[t] = V_RS / max([mu_0, threshold])**2
            rV_H[:, t] = V_H / max([mu_0, threshold])**2
            print('Relative Variance of Random Sampling: {:f}'.format(rV_RS[t]))
            print('Relative Variance of Hashing schmes')
            print(rV_H[:, t])
        # estimate the mean number of samples required by each method
        rV_RS_mean = np.mean(rV_RS)
        rV_H_mean = np.mean(rV_H, 1)

        # use the log-condition plot to visualize local query structure for
        # the dataset
        if visualization_flag:
            plt.figure()
            plt.bar(np.arange(len(collision_prob)+1), \
                    np.append(rV_RS_mean, rV_H_mean), \
                    tick_label=map(lambda x: str(x), \
                                   ['RS'] + range(len(collision_prob))))
            plt.title('Relative Variance for Random Queries')

            self.log_condition_plot(lambda_eps, Lambda_eps, threshold)

        return (rV_RS_mean, rV_H_mean)

#%% Demo that implements the approach of Siminelakis, Rong, Bailis, Charikar
#   Levis, ICML 2019.
if __name__ == "__main__":
    #%%  Problem Specificaiton
    # example dataset
    points = np.loadtxt(open("datasets/covtype.data", "rb"), delimiter=",",\
                        skiprows=0)
    points = points.transpose()
    (d,n) = points.shape
    # specify kernel evaluation paramaters
    weights = np.ones(n) / n
    sigma = np.sqrt(np.mean(np.mean(points**2))) / 5 # example bandwidth
    # exponential kernel
    kernel_fun = lambda x,y: np.exp(-np.linalg.norm(x-y) / sigma)

    #%%  Rehashing methodology
    # lower bound of densities of interest suggested value is 0.1 / sqrt{n}
    tau = 0.0001
    # target multiplicative accuracy
    eps = 0.2
    # Initialize data structure
    covtype = rehashing(points, weights, kernel_fun)
    # Create sketch
    covtype.create_sketch(method='random', accuracy=eps, threshold=tau,\
                  num_of_means=1)
    # Set parameters for hashing through eLSH
    hash_probs = []
    R = np.log(1/eps/tau) # effective diameter for exponential kernel

    # For illustration we will apply our method for 2 hashing schemes
    # but one can extend this to an arbitrary number of hashing schemes

    # Hashing scheme #1
    kappa_0 = int(np.ceil(np.sqrt(2*np.pi)*R*np.log(1/tau))) # ``power"
    w0 = np.sqrt(2 / np.pi) * 2 * kappa_0 # set hash bucket width
    p0 = partial(eLSH_prob, w=w0*sigma, k=kappa_0) # collision probability
    hash_probs.append(p0) # add to list

    # Hashing scheme #2
    kappa_1 = kappa_0 / 5 # ``power"
    w1 = np.sqrt(2 / np.pi) * 2 * kappa_1 # set hash bucket width
    p1 = partial(eLSH_prob, w=w1*sigma, k=kappa_1)  # collision probability
    hash_probs.append(p1) # add to list

    # Run diagnostic and visualization
    covtype.diagnostic(hash_probs, acc=eps, threshold=tau, \
                   num_queries=100, visualization_flag=True)

    # Use the diagnostic procedure to specify a config file for C++ code.
