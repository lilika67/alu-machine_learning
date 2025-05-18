#!/usr/bin/env python3
"""
Bidirectional Cell Module
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN.
    """

    def __init__(self, i, h, o):
        """
        Initializes the bidirectional cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        # Weights for the forward direction
        self.Whf = np.random.randn(h + i, h)
        self.bhf = np.zeros((1, h))

        # Weights for the backward direction
        self.Whb = np.random.randn(h + i, h)
        self.bhb = np.zeros((1, h))

        # Weights for the output
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Input data for the cell of shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state.
        """
        # Concatenate h_prev and x_t
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state
        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)

        return h_next
