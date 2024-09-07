#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Data for the plot
y = np.arange(0, 11) ** 3

# Plotting y as a red line
plt.plot(y, color='red', linestyle='-')

# Setting the x-axis limits
plt.xlim(0, 10)

# Displaying the plot
plt.show()

