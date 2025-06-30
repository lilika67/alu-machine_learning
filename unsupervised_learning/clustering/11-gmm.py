#!/usr/bin/env python3
"""
Defines function that calculates a GMM from a dataset
"""


import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            contains the number of clusters

    returns:
        pi, m, S, clss, bic:
            pi [numpy.ndarray of shape (k,)]:
                containing the cluster priors
            m [numpy.ndarray of shape (k, d)]:
                containing the centroid means
            S [numpy.ndarray of shape (k, d, d)]:
                containing the covariance matrices
            clss [numpy.ndarray of shape (n,)]:
                containting the cluster indices for each data point
            bic [numpy.ndarray of shape (kmax - kmin + 1)]:
                containting the BIC value for each cluster size tested
    """
    # Fit the Gaussian Mixture Model to the data
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    # Extract the required parameters
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
