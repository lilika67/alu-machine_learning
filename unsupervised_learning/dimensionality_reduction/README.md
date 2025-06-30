# Dimensionality Reduction using PCA

This project focuses on implementing Principal Component Analysis (PCA) for dimensionality reduction. PCA is a widely used technique in machine learning and data analysis to reduce the number of features (dimensions) in a dataset while preserving as much variance as possible. The project consists of two tasks:

1. **PCA with Variance Retention**: Implement PCA to reduce the dimensionality of a dataset while retaining a specified fraction of the original variance.
2. **PCA with Fixed Dimensionality**: Implement PCA to reduce the dimensionality of a dataset to a fixed number of dimensions.

---

## Requirements

### General
- Allowed editors: `vi`, `vim`, `emacs`
- All files will be interpreted/compiled on **Ubuntu 16.04 LTS** using **Python 3.5**.
- All files must end with a new line.
- The first line of all files must be exactly `#!/usr/bin/env python3`.
- A `README.md` file at the root of the project folder is mandatory.
- Code must follow the `pycodestyle` style (version 2.4), with no more than 79 characters per line (including spaces).
- All modules, classes, and functions must have documentation.
- Only the `numpy` library is allowed for import (as `import numpy as np`).
- All files must be executable.

### Data
The project uses the following datasets for testing:
- `mnist2500_X.txt`: A dataset of 2500 samples with 784 features (e.g., flattened MNIST images).
- `mnist2500_labels.txt`: Labels corresponding to the dataset.

---

## Tasks

### Task 0: PCA with Variance Retention
#### Description
Implement a function `pca(X, var=0.95)` that performs PCA on a dataset `X` and retains a specified fraction of the original variance.

- **Input**:
  - `X`: A `numpy.ndarray` of shape `(n, d)`, where `n` is the number of data points and `d` is the number of dimensions.
  - `var`: A float representing the fraction of variance to retain (default is 0.95).
- **Output**:
  - `W`: A `numpy.ndarray` of shape `(d, nd)`, where `nd` is the new dimensionality of the transformed data.

#### Implementation
The function uses Singular Value Decomposition (SVD) to compute the principal components. It calculates the cumulative explained variance and selects the minimum number of components required to retain the specified variance.

#### Example Usage
```python
import numpy as np
pca = __import__('0-pca').pca

# Generate synthetic data
np.random.seed(0)
X = np.random.normal(size=(100, 10))
X_centered = X - np.mean(X, axis=0)

# Perform PCA
W = pca(X_centered, var=0.95)
print(W.shape)  # Output: (10, nd), where nd is the number of components needed to retain 95% variance
```

---

### Task 1: PCA with Fixed Dimensionality
#### Description
Implement a function `pca(X, ndim)` that performs PCA on a dataset `X` and reduces its dimensionality to a fixed number of dimensions.

- **Input**:
  - `X`: A `numpy.ndarray` of shape `(n, d)`, where `n` is the number of data points and `d` is the number of dimensions.
  - `ndim`: An integer representing the new dimensionality of the transformed data.
- **Output**:
  - `T`: A `numpy.ndarray` of shape `(n, ndim)` containing the transformed version of `X`.

#### Implementation
The function centers the data, performs SVD, and selects the top `ndim` principal components to transform the data into the lower-dimensional space.

#### Example Usage
```python
import numpy as np
pca = __import__('1-pca').pca

# Load dataset
X = np.loadtxt("mnist2500_X.txt")

# Perform PCA
T = pca(X, ndim=50)
print(T.shape)  # Output: (2500, 50)
```

---

## Repository Structure
```
dimensionality_reduction/
├── 0-pca.py              # Task 0: PCA with variance retention
├── 1-pca.py              # Task 1: PCA with fixed dimensionality
├── 0-main.py             # Test file 0
├── 1-main.py             # Test file 1
├── README.md             # Project documentation
├── mnist2500_X.txt       # Dataset (features)
└── mnist2500_labels.txt  # Dataset (labels)
```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/alu-machine_learning.git
   cd alu-machine_learning/unsupervised_learning/dimensionality_reduction
   ```
2. Run the scripts:
   - For Task 0:
     ```bash
     ./0-main.py
     ```
   - For Task 1:
     ```bash
     ./1-main.py
     ```

---

## Notes
- Ensure that the dataset files (`mnist2500_X.txt` and `mnist2500_labels.txt`) are in the same directory as the scripts.
- The scripts are designed to work with Python 3.5 and require the `numpy` library.

---

## Author
- **Maxime Guy Bakunzi** - [GitHub Profile](https://github.com/Maxime-Bakunzi)

