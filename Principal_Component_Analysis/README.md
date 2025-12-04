# 📊 Principal Component Analysis (PCA) Implementation

A C++ implementation of Principal Component Analysis (PCA), along with a Python wrapper for easy execution and parsing of results, uses the Eigen library for efficient linear algebra.

## 📝 Overview

The C++ program (`pca`) performs PCA on a given input data matrix (in Matrix Market format) by:
1.  Reading the input data matrix.
2.  Centering the data (subtracting the column means).
3.  Computing the Singular Value Decomposition (SVD) of the centered data matrix.
4.  Calculating the Principal Component Scores (transformed data).
5.  Calculating the Loadings (principal directions).
6.  Reporting the Singular Values and Explained Variance Ratios.
7.  Writing the Principal Component Scores to an output file.

The Python script (`pca_wrapper.py`) provides a convenient interface to execute the C++ binary and load the results (scores, singular values, and explained variance) directly into NumPy arrays.

## 🛠️ Prerequisites

To build and run the C++ program, you need:

A C++ compiler that supports C++20 (e.g., `g++` or `clang++`).
The Eigen 3 library (development files). The `Makefile` assumes it's installed at `/usr/include/eigen3`.
OpenMP (optional, for parallelization, included in `g++` on most systems).

To use the Python wrapper, you need:

Python 3.x
NumPy

## 🏗️ Building the C++ Program

Use the provided `Makefile` to compile the C++ source file (`pca.cpp`).

```bash
make
````

### Build Details

The `Makefile` uses the following settings:

`CXX = g++`
`CXXFLAGS = -std=c++20 -O3 -march=native -fopenmp -DNDEBUG -I/usr/include/eigen3 -Wall -Wextra -Wno-maybe-uninitialized`
The output executable is named `pca`.

To clean up the compiled executable and output files:

```bash
make clean
```

## 💾 Data Format (Matrix Market)

Both the input (`A.dat`) and output (`PC_scores.dat`) files use the Matrix Market Coordinate Format for real, general matrices.

### Input Data (`A.dat`)

The input file must contain the data matrix $\mathbf{A}$, where:

Rows represent samples ($N$).
Columns represent features ($M$).

The format is:

```
%%MatrixMarket matrix coordinate real general
<rows> <cols> <non_zero_entries>
<row_index> <col_index> <value>
...
```

### Example Input (`A.dat`):

```matrix-market
%%MatrixMarket matrix coordinate real general
5 4 19
1 1 1.0
...
5 4 -1.0
```

## 🚀 Usage

### C++ Executable (`./pca`)

The C++ program expects the input file to be named `A.dat` in the current working directory and writes the output to `PC_scores.dat`.

```bash
./pca
```

Output to STDOUT (Example):
The program prints information about thread usage, timings, and the calculated PCA components:

```
Using 4 OpenMP threads for Eigen.
[Timer] Reading A.dat took: 0.XX ms.
Data matrix A loaded: 5 samples x 4 features

[Timer] SVD computation took: 0.YY ms.
...
Singular values: [s_1, s_2, s_3, ...]

Explained variance ratio: [evr_1, evr_2, evr_3, ...]
Cumulative explained variance: [cum_1, cum_2, cum_3, ...]
Matrix rank: R
Principal component scores written to PC_scores.dat
```

### Python Wrapper (`pca_wrapper.py`)

The `pca_wrapper.py` script simplifies running the C++ program and handles data parsing.

It contains the function `run_cpp_pca` which executes the C++ binary and returns a dictionary of results.

```python
import numpy as np
from pca_wrapper import run_cpp_pca

# Ensure 'pca' is built and 'A.dat' is present in the current directory
try:
    result = run_cpp_pca(
        executable="./pca",
        a_dat="A.dat",
        output_scores="PC_scores.dat",
        cwd="."
    )

    scores = result["scores"]
    explained_variance_ratio = result["explained_variance_ratio"]
    singular_values = result["singular_values"]

    print("Scores shape:", scores.shape)
    print("Explained Variance Ratio:", explained_variance_ratio)

except Exception as e:
    print(f"An error occurred during PCA execution: {e}")
```

### `run_cpp_pca` Return Dictionary

The `run_cpp_pca` function returns a dictionary with the following keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `scores` | `np.ndarray` | The Principal Component Scores matrix (Samples x Components). |
| `explained_variance_ratio` | `np.ndarray` or `None` | The ratio of variance explained by each component. |
| `singular_values` | `np.ndarray` or `None` | The singular values ($\mathbf{s}$) from the SVD. |
| `loadings` | `None` | Not currently parsed from the output files/stdout. |
| `stdout` | `str` | The complete console output from the C++ program. |

## 📐 Mathematical Basis

The program performs PCA by using the Singular Value Decomposition (SVD) on the centered data matrix $\mathbf{A}_c$.

$$\mathbf{A}_c = \mathbf{U} \mathbf{S} \mathbf{V}^T$$

$\mathbf{A}_c$: The input data matrix, centered ($\text{samples} \times \text{features}$).
$\mathbf{V}$: The Loadings or Principal Directions ($\text{features} \times \text{components}$). This is `loadings` in `pca.cpp`.
$\mathbf{S}$: A diagonal matrix containing the Singular Values ($\mathbf{s}$).
$\mathbf{U}$: The $\text{left-singular vectors}$.

The Principal Component Scores ($\mathbf{T}$) are the transformed data:

$$\mathbf{T} = \mathbf{A}_c \mathbf{V}$$

$\mathbf{T}$: The Principal Component Scores ($\text{samples} \times \text{components}$). This is `scores` in `pca.cpp` and written to `PC_scores.dat`.

The Explained Variance Ratio for component $k$ is proportional to the square of the corresponding singular value $s_k^2$:

$$\text{Explained Variance Ratio}_k = \frac{s_k^2}{\sum_i s_i^2}$$

