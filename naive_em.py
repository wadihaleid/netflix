"""Mixture model using EM"""
from typing import Tuple
from matplotlib.pyplot import pie
import numpy as np
from numpy.core.arrayprint import _none_or_positive_arg
from numpy.lib.scimath import power
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # calculate gaussian for each K
    variances = mixture.var     
    mus = mixture.mu    
    props = mixture.p
    k = len(variances)
    #  c = calculate_gaussian_constant(sigma , k)
    f = np.zeros([X.shape[0] , k])
    N = np.zeros([X.shape[0] , k])
    post = np.zeros([X.shape[0] , k])
    total_prop = np.zeros(X.shape[0])

    for i in range(k):
        mu = mus[i,:]
        for j in range(X.shape[0]) :
            dev = X[j,:] - mu
            var_ii = variances[i]
            e_ = np.math.exp((-1/(2*var_ii)) * dev.dot(dev))
            N[j,i] = (1/(np.lib.scimath.power(2*3.141592653589793*var_ii ,X.shape[1]/2))) * e_
            f[j,i] = props[i] * N[j,i]

    j = 0
    for i in f:
        total_prop[j] = sum(i)
        j += 1
    
    for j in range(f.shape[1]):
        post[:,j] = f[:,j] / total_prop

    max_likelihood_proxy = 0.0
    for i in range(post.shape[0]):
        for j in range(post.shape[1]):
            r = np.log(f[i,j]) - np.log(post[i,j])
            max_likelihood_proxy = max_likelihood_proxy + post[i,j] * r


    max_likelihood_actual = 0.0
    
    for i in range(f.shape[0]):
        s = 0
        for j in range(f.shape[1]):
            s = s + f[i,j]
        max_likelihood_actual = (max_likelihood_actual + np.log(s) )

    return [post , max_likelihood_proxy]



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    # Calculate p, mu and variance for each cluster. 

    p = np.zeros(post.shape[1])
    mu = np.zeros([post.shape[1] , X.shape[1]])
    var_ = np.zeros(post.shape[1])

    n = X.shape[0]
    d = X.shape[1]

    for j in range(post.shape[1]):
        p[j] = sum(post[:,j]) / n

    # New mu 
    for j in range(post.shape[1]):
        for i in range(X.shape[0]): 
            mu[j,:] = mu[j,:] + X[i,:] * post[i,j]
        mu[j,:] = mu[j,:] / sum(post[:,j])
    
    # New Variances
    for j in range(post.shape[1]):
        for i in range(X.shape[0]): 
            dev = X[i,:] - mu[j,:]
            var_[j] = var_[j] + (dev.dot(dev) * post[i,j])
        var_[j] = var_[j] / (d*sum(post[:,j]))

    return GaussianMixture(mu , var_ , p)   



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = 0 
    new_ll = 0

    new_post , new_ll = estep(X, mixture)
    while (np.abs(new_ll - old_ll) > 0.000001 * np.abs(new_ll)):
        old_ll = new_ll
        new_mixture = mstep(X , new_post)
        new_post , new_ll = estep(X, new_mixture)
    
    return new_mixture , new_post , new_ll



