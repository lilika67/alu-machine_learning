#!/usr/bin/env python3
"""
Defines function that performs K-means on a dataset
"""


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            contains the number of clusters
        iterations [positive int]:
            contains the maximum number of iterations that should be performed

    if no change in the cluster centroids occurs between iterations,
        the function should return

    initialize the cluster centroids using a multivariate unitform distribution

    if a cluster contains no data points during the update step,
        its centroid should be reinitialized

    should use:
        numpy.random.uniform exactly twice
        at most 2 loops

    returns:
        C, clss:
            C [numpy.ndarray of shape (k, d)]:
                containing the centroid means for each cluster
            clss [numpy.ndarray of shape (n,)]:
                containting the index of the cluster in c
                    that each data point belongs to
        or None, None on failure
    """
    if type(X) is not np.ndarray or type(k) is not int:
        return (None, None)
    if len(X.shape) != 2 or k < 0:
        return (None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None)
    n, d = X.shape
    if k == 0:
        return (None, None)
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
        new_C = np.copy(C)
        for c in range(k):
            if c not in clss:
                new_C[c] = np.random.uniform(low, high)
            else:
                new_C[c] = np.mean(X[clss == c], axis=0)
        if (new_C == C).all():
            return (C, clss)
        else:
            C = new_C
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=-1), axis=-1)
    return (C, clss)
