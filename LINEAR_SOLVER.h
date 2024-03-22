#ifndef _LINEAR_SOLVER_H_
#define _LINEAR_SOLVER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string.h>              // required for memcpy()
#include <float.h>               // required for DBL_EPSILON
#include <math.h>                // required for fabs(), sqrt();
using namespace std;

// Macro for accessing elements of a matrix (assuming A is the matrix)
#define a(i, j) A[i][j]

// Template for a 2D matrix using std::vector
template<typename T>
using matrix = std::vector<std::vector<T>>;

// Function prototypes for linear solvers and file I/O functions
int SOR(const matrix<double>& A, vector<double>& B, vector<double>& X, unsigned int num_rows, unsigned int num_cols, double tolerance, unsigned int Max_Iter);
int Gauss_Seidel(const matrix<double>& A, vector<double>& B, vector<double>& X, unsigned int num_rows, unsigned int num_cols, double tolerance, unsigned int Max_Iter);
int Jacobi(const matrix<double>& A, vector<double>& B, vector<double>& X, unsigned int num_rows, unsigned int num_cols, double tolerance, unsigned int Max_Iter);
int ReadMatrixMM(matrix<double>& A, unsigned int& num_rows, unsigned int& num_cols);
int ReadVectorMM(vector<double>& B, unsigned int& num_cols);
int WriteMatrixMM(const matrix<double>& A, unsigned int num_rows, unsigned int num_cols);
int WriteVectorMM(const std::vector<double>& X, unsigned int num_rows, unsigned int num_cols);

// Function prototypes for printing and creating vectors/matrices
void PrintVector(const std::vector<double>& vector);
void PrintVector2D(const matrix<double>& A);
std::vector<double> CreateZeroVector(unsigned int num_rows);

#endif /* _LINEAR_SOLVER_H_ */
