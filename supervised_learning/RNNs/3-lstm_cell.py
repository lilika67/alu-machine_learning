#!/usr/bin/env python3
"""
LSTM Cell Module
"""

import numpy as np


class LSTMCell:
    """
    Represents a cell of a Long Short-Term Memory (LSTM) network.
    """

    def __init__(self, i, h, o):
        """
        Initializes the LSTM cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        # Weights for the forget gate
        self.Wf = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))

        # Weights for the update gate
        self.Wu = np.random.randn(h + i, h)
        self.bu = np.zeros((1, h))

        # Weights for the intermediate cell state
        self.Wc = np.random.randn(h + i, h)
        self.bc = np.zeros((1, h))

        # Weights for the output gate
        self.Wo = np.random.randn(h + i, h)
        self.bo = np.zeros((1, h))

        # Weights for the output
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            c_prev (numpy.ndarray): Previous cell state of shape (m, h).
            x_t (numpy.ndarray): Input data for the cell of shape (m, i).

        Returns:
            h_next (numpy.ndarray): Next hidden state.
            c_next (numpy.ndarray): Next cell state.
            y (numpy.ndarray): Output of the cell.
        """
        # Concatenate h_prev and x_t
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = self.sigmoid(np.dot(concat_input, self.Wf) + self.bf)

        # Update gate
        u = self.sigmoid(np.dot(concat_input, self.Wu) + self.bu)

        # Intermediate cell state
        c_tilde = np.tanh(np.dot(concat_input, self.Wc) + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_tilde

        # Output gate
        o = self.sigmoid(np.dot(concat_input, self.Wo) + self.bo)

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y

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
