#!/usr/bin/env python3
"""
RNN Module
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Args:
        rnn_cell (RNNCell): Instance of RNNCell used for forward propagation.
        X (numpy.ndarray): Input data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state of shape (m, h).

    Returns:
        H (numpy.ndarray): All hidden states of shape (t + 1, m, h).
        Y (numpy.ndarray): All outputs of shape (t, m, o).
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    # Initialize arrays to store hidden states and outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set the initial hidden state
    H[0] = h_0

    # Perform forward propagation for each time step
    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
