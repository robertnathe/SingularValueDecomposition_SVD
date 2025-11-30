Matrix Market SVD Solver

The C++ program reads a matrix A and a vector B from files in Matrix Market coordinate format, solves the linear system A·X = B using Singular Value Decomposition (SVD), and writes the solution vector X to the disk. The program uses the Eigen library for linear algebra operations.

Features

    Reads matrices and vectors from Matrix Market coordinate files (A.dat, B.dat)

    Solves linear systems using SVD (robust to rank-deficient or ill-conditioned matrices)

    Prints solution vector and singular values to the console

    Writes the solution vector to a Matrix Market file (X_out.dat)

    Reports execution time for the SVD solve step

    Includes utility functions for pretty-printing matrices and vectors

Requirements

    C++20 or newer

    Eigen library (header-only)

    Standard C++ libraries (iostream, fstream, vector, etc.)


