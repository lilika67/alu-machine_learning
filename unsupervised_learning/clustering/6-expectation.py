#!/usr/bin/env python3
"""
Defines function that calculates the expectation step in the EM algorithm
for a Gaussian Mixture Model
"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        pi [numpy.ndarray of shape (k,)]:
            contains the priors for each cluster
        m [numpy.ndarray of shape (k, d)]:
            contains the centroid means for each clustern
        S [numpy.ndarray of shape (k, d, d)]:
            contains the covariance matrices for each cluster

    should only use one loop

    returns:
        g, l:
            g [numpy.ndarray of shape (k, n)]:
                containing the posterior probabilities for each data point
                    in the cluster
            l [float]:
                total log likelihood
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        g = np.zeros((k, n))
        for i in range(k):
            pdf_result = pdf(X, m[i], S[i])
            if pdf_result is None:
                return None, None
            g[i] = pi[i] * pdf_result

        total_likelihood = np.sum(g, axis=0)
        if np.any(total_likelihood == 0):
            return None, None

        likelihood = np.sum(np.log(total_likelihood))
        g /= total_likelihood
        return g, likelihood
    except Exception as e:
        return None, None
