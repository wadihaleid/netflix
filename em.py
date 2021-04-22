"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    variances = mixture.var     
    mus = mixture.mu    
    props = mixture.p
    k = len(variances)    
    f = np.zeros([X.shape[0] , k])
    N = np.zeros([X.shape[0] , k])
    post = np.zeros([X.shape[0] , k])
    total_prop = np.zeros(X.shape[0])

    for i in range(k):
        mu = mus[i,:]
        for j in range(X.shape[0]) :  
            var_ii = variances[i]
            dev = np.zeros(X.shape[1])
            for u in range(X.shape[1]):
                if (X[j,u] != 0):
                    dev[u] = X[j,u] - mu[u]
                else:
                    dev[u] = 0
            e_ = np.math.exp((-1/(2*var_ii)) * dev.dot(dev))
            degree =  sum(X[j,:] > 0)
            N[j,i] = float((1/(np.lib.scimath.power(2*3.141592653589793*var_ii , degree/2))) * e_)
            f[j,i] = props[i]*N[j,i]

    j = 0
    for i in f:
        total_prop[j] = sum(i)
        j += 1
    
    for j in range(f.shape[1]):
        post[:,j] = f[:,j]/total_prop

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



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    p = np.zeros(post.shape[1])
    mu = np.zeros([post.shape[1] , X.shape[1]])
    var_ = np.zeros(post.shape[1])

    n = X.shape[0]
    d = X.shape[1]

    for j in range(post.shape[1]):
        p[j] = sum(post[:,j]) / n

    # New mu 
    for j in range(post.shape[1]): ## cluster loop       
        for l in range(X.shape[1]): ## data column loop
            s = 0
            total_s = 0
            for u in range(X.shape[0]): # data loop (observations)
                m = post[u,j] * X[u,l] 
                s = s + m
                if (X[u,l] != 0):
                    total_s = total_s + post[u,j]
            if (total_s >= 1):
                mu[j,l] = s / total_s
            else:
                mu[j,l] = mixture.mu[j,l]


    # New Variances
    for j in range(post.shape[1]):                
        s = 0        
        s_total = 0
        for u in range(X.shape[0]): # data loop (observations)
            dev = np.zeros(mu.shape[1])
            c = 0
            for l in range(X.shape[1]): ## data column loop
                if (X[u,l] != 0):
                    dev[l] = X[u,l] - mu[j,l]
                    c += 1
                else:
                    dev[l] = 0
            s = s + post[u , j] * (dev.dot(dev))
            s_total = s_total + post[u,j] * c
        if ((s / s_total) > min_variance):
            var_[j] = s / s_total
        else:
            var_[j] = min_variance

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
        new_mixture = mstep(X , new_post , mixture)
        new_post , new_ll = estep(X, new_mixture)
        mixture = new_mixture
    
    return new_mixture , new_post , new_ll




def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    X_completed = np.zeros([X.shape[0] , X.shape[1]])
    variances = mixture.var     
    mus = mixture.mu    
    props = mixture.p
    k = len(variances)    
    f = np.zeros([X.shape[0] , k])
    N = np.zeros([X.shape[0] , k])
    post = np.zeros([X.shape[0] , k])
    total_prop = np.zeros(X.shape[0])

    for i in range(k):
        mu = mus[i,:]
        for j in range(X.shape[0]) :  
            var_ii = variances[i]
            dev = np.zeros(X.shape[1])
            for u in range(X.shape[1]):
                if (X[j,u] != 0):
                    dev[u] = X[j,u] - mu[u]
                else:
                    dev[u] = 0
            e_ = np.math.exp((-1/(2*var_ii)) * dev.dot(dev))
            degree =  sum(X[j,:] > 0)
            N[j,i] = float((1/(np.lib.scimath.power(2*3.141592653589793*var_ii , degree/2))) * e_)
            f[j,i] = props[i]*N[j,i]

    j = 0
    for i in f:
        total_prop[j] = sum(i)
        j += 1
    
    for j in range(f.shape[1]):
        post[:,j] = f[:,j]/total_prop
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if (X[i,j] == 0):
                max_i = np.argmax(post[i,:])
                p = np.max(post[i,:])
                X_completed[i,j] = mus[:,j].dot(post[i,:])
            else:
                X_completed[i,j] = X[i,j]

    return X_completed