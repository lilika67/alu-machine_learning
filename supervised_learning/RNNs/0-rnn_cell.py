#!/usr/bin/env python3
"""
RNN Cell Module
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.
    """

    def __init__(self, i, h, o):
        """
        Initializes the RNN cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Input data for the cell of shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state.
            y (numpy.ndarray): Output of the cell.
        """
        # Concatenate h_prev and x_t
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)

        # Compute the output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, x):
        """
        Computes the softmax of x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Softmax of x.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
