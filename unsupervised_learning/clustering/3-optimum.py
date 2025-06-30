#!/usr/bin/env python3
"""
Defines function that tests for the optimum number of clusters by variance
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        kmin [positive int]:
            containing the minimum number of clusters to check for (inclusive)
        kmax [positive int]:
            containing the maximum number of clusters to check for (inclusive)
        iterations [positive int]:
            containing the maximum number of iterations for K-means

    function should analyze at least 2 different cluster sizes

    should use at most 2 loops

    returns:
        results, d_vars:
            results [list]:
                containing the output of K-means for each cluster size
            d_vars [list]:
                containing the difference in variance from the smallest cluster
                    size for each cluster size
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]

    results = []
    d_vars = []
    var = float('inf')
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        new_var = variance(X, C)
        if k == kmin:
            var = new_var
        d_vars.append(var - new_var)
    return results, d_vars
