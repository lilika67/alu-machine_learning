#!/usr/bin/env python3
"""
GRU Cell Module
"""

import numpy as np


class GRUCell:
    """
    Represents a cell of a Gated Recurrent Unit (GRU).
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRU cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        # Weights for the update gate
        self.Wz = np.random.randn(h + i, h)
        self.bz = np.zeros((1, h))

        # Weights for the reset gate
        self.Wr = np.random.randn(h + i, h)
        self.br = np.zeros((1, h))

        # Weights for the intermediate hidden state
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))

        # Weights for the output
        self.Wy = np.random.randn(h, o)
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

        # Update gate
        z = self.sigmoid(np.dot(concat_input, self.Wz) + self.bz)

        # Reset gate
        r = self.sigmoid(np.dot(concat_input, self.Wr) + self.br)

        # Intermediate hidden state
        h_tilde = np.tanh(np.dot(np.concatenate(
            (r * h_prev, x_t), axis=1), self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y

    def sigmoid(self, x):
        """
        Computes the sigmoid of x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid of x.
        """
        return 1 / (1 + np.exp(-x))

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
