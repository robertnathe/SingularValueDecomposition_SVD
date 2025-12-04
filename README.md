Singular Value Decomposition (SVD) Solver: C++ Core with Python Wrapper

The Singular Value Decomposition (SVD) algorithm solves the linear least squares problem A x = b using a high-performance C++ core (built with the Eigen library) for the computational logic, complemented by a convenient Python wrapper for streamlined data handling and execution. 

Features

Robust Solver: Solves the linear system A x = b using SVD, which is robust even for rank-deficient or ill-conditioned matrices.
High Performance C++ Backend: The SVD calculation is performed by a C++ executable (`Singular_Value_Decomposition_SVD`) utilizing the Eigen library (`BDCSVD` or `JacobiSVD`) for optimized linear algebra operations.
Standardized I/O: The C++ program accepts input matrix $\mathbf{A}$ and vector $\mathbf{b}$ via files (`A.dat`, `B.dat`) in the Matrix Market Coordinate format.
Python Integration: A Python class, `SVDWrapper`, provides an interface to compile the C++ code, write NumPy arrays to the required input files, execute the C++ solver, and read the solution vector x back into a NumPy array.

Prerequisites

C++ Backend (`Singular_Value_Decomposition_SVD.cpp`)

To build and run the C++ executable, you need:

1.  C++ Compiler: A C++20 compatible compiler (e.g., `g++` or `clang++`).
2.  Build Tool: `make`.
3.  Eigen Library: The Eigen C++ template library for linear algebra. You must ensure the header files are accessible to the compiler
    e.g., installed at `/usr/include/eigen3`).

Python Wrapper (`svd_wrapper.py`)

To use the Python wrapper, you need:

1.  Python 3.x
2.  NumPy
3.  Required System Utilities: The wrapper uses `subprocess` to call `make` and the compiled C++ executable, so these must be available in your system's PATH.

Building and Usage

1\. Building the C++ Executable

The `Makefile` is provided to simplify compilation.

```bash
# Compile the C++ program
make
```

This command will create the executable file named `Singular_Value_Decomposition_SVD`.

2\. Usage via Python Wrapper (Recommended)

The `svd_wrapper.py` file contains the `SVDWrapper` class, which handles the entire workflow of compilation, I/O, and execution using NumPy arrays.

```python
import numpy as np
from svd_wrapper import SVDWrapper

# 1. Initialize the solver
solver = SVDWrapper()

# 2. Compile the C++ code (first time setup)
try:
    solver.compile()
except Exception as e:
    print(f"Compilation failed: {e}")
    # Handle error

# 3. Define the system A*x = b using NumPy arrays
A_matrix = np.array([
    [1.0, 1.0, 0.0, 3.0],
    [2.0, 1.0, -1.0, 1.0],
    [3.0, -1.0, -1.0, 2.0],
    [-1.0, 2.0, 3.0, -1.0],
    [-1.0, 2.0, 3.0, -1.0]
])

b_vector = np.array([4.0, 1.0, -3.0, 4.0, 2.0])

# 4. Solve the system
try:
    x_solution = solver.solve(A_matrix, b_vector)
    print("\nSolution x:\n", x_solution)
except Exception as e:
    print(f"Solving failed: {e}")

# Optional: Clean up temporary files (A.dat, B.dat, X_out.dat)
# solver.clean()
```

The `solver.solve(A, b)` method automatically writes the input matrix $\mathbf{A}$ to `A.dat` and vector b to `B.dat`, runs the C++ executable, and reads the result x from `X_out.dat`.

3\. Usage via C++ Executable (Standalone)

For standalone C++ execution:

1.  Ensure you have compiled the executable (`make`).
2.  Manually create the input files `A.dat` (matrix $\mathbf{A}$) and `B.dat` (vector b) in the Matrix Market Coordinate format.
3.  Run the executable:
    ```bash
    ./Singular_Value_Decomposition_SVD
    ```
4.  The program will output the solution vector $\mathbf{x}$, singular values, and execution time to the console.
5.  The solution vector $\mathbf{x}$ will also be saved to a file named `X_out.dat`.

Input and Output File Format

The C++ program is designed to read and write files based on the Matrix Market format standards.

| File | Role | Format | Details |
| :--- | :--- | :--- | :--- |
| `A.dat` | Input Matrix $\mathbf{A}$ | Matrix Market Coordinate (Real General) | Contains rows, columns, and non-zero entries for $\mathbf{A}$. |
| `B.dat` | Input Vector $\mathbf{b}$ | Matrix Market Coordinate (Real General) | Formatted as a column vector (e.g., $M \times 1$). |
| `X_out.dat` | Output Vector $\mathbf{x}$ | Modified Matrix Market Array | Contains the calculated solution vector $\mathbf{x}$. |

The Python `SVDWrapper` handles the translation between NumPy arrays and these file formats, including managing the 1-based indexing expected by the C++ parser.

License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 (AGPLv3). Please see the `LICENSE` file for full terms and conditions.
