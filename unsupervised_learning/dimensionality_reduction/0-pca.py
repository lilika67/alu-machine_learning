#!/usr/bin/env python3
"""
Defines the function that performs PCA on a dataset with zero mean.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset with zero mean.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) with zero mean.
        var (float): Fraction of variance to retain. Defaults to 0.95.

    Returns:
        numpy.ndarray: Weights matrix of shape (d, nd).
    """
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(X)

    # Calcuate the varience ratios
    ratios = list(x / np.sum(S) for x in S)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0]

    # Construct the weight matrix
    W = Vt.T[:, :(nd + 1)]
    return (W)
