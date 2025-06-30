#!/usr/bin/env python3
"""
Defines function that calculates the probability density function of
a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset whose PDF should be calculated
            n: the number of data points
            d: the number of dimensions for each data point
        m [numpy.ndarray of shape (d,)]:
            contains the mean of the distribution
        S [numpy.ndarray of shape (d, d)]:
            contains the covariance of the distribution

    not allowed to use any loops
    not allowed to use the function numpy.diag or method numpy.ndarray.diagonal

    returns:
        P [numpy.ndarray of shape (n,)]:
            containing the PDF values for each data point
            all values in P should have a minimum value of 1e-300
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or d != S.shape[0] or d != S.shape[1]:
        return None
    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    fac = 1 / np.sqrt(((2 * np.pi) ** d) * S_det)
    X_m = X - m
    X_m_dot = np.dot(X_m, S_inv)
    X_m_dot_X_m = np.sum(X_m_dot * X_m, axis=1)
    P = fac * np.exp(-0.5 * X_m_dot_X_m)
    return np.maximum(P, 1e-300)
