#!/usr/bin/env python3
"""
Defines the function that perfoms PCA on a dataset with dimensionality.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        ndim (int): New dimensionality of the transformed data.

    Returns:
        numpy.ndarray: Transformed data of shape (n, ndim).
    """
    # Center the data by subtracting the mean of each feature
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform Singular Value Decomposition on the centered data
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Select the top 'ndim' principal components (rows of Vt)
    W = Vt[:ndim].T

    # Transform the data by projecting onto the principal components
    T = np.dot(X_centered, W)

    return T
