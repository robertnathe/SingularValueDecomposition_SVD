#ifndef _SVD_SOLVER_H_
#define _SVD_SOLVER_H_

#include <string.h>              // required for memcpy()
#include <float.h>               // required for DBL_EPSILON
#include <math.h>                // required for fabs(), sqrt();
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std; 
#define a(i, j) A[i][j] // Assuming A is a matrix (use appropriate access method)
template<typename T>
using matrix = std::vector<std::vector<T>>;

void print_1DFormatMatrix(const double data[], int num_rows, int num_cols); 
void print_1DFormatVector(const double data[], int num_cols);
int ReadMatrixMM(matrix<double> & A, int& num_rows, int& num_cols); // For matrices 
int ReadVectorMM(vector<double>& B, int& num_cols);
int WriteMatrixMM(const matrix<double>& A, unsigned int num_rows, unsigned int num_cols);
int WriteVectorMM(const std::vector<double>& X, unsigned int num_rows, unsigned int num_cols);
void PrintVector(const std::vector<double>& vector);
void PrintVector2D(const matrix<double>& A);
void print_matrix (double *X, int num_rows, int num_cols);
void Zero_Matrix(double* X, int num_rows, int num_cols);
void Zero_Vector(double* X, int num_cols);
int print_output(int num_rows, int num_cols, int num_rows_B, int num_cols_B,
                 const double array_U[], const double D[], const double array_V[], const vector<double>& X);
int Parameters_Solve (matrix<double> & AMatrix, int num_rows, int num_cols, double array_U[], double D[], double *V, double B[], vector<double>& X);
int Singular_Value_Decomposition(matrix<double> & AMatrix, double* A, int num_rows, int num_cols, double* U, double* singular_values, double* V, double* dummy_array);
void Singular_Value_Decomposition_Solve(double* U, double* D, double* V, double tolerance, int num_rows, int num_cols, double *B, vector<double>& X);
void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V, double tolerance, int num_rows, int num_cols, double *A_Pseudo_Inverse);

#endif /* _SVD_SOLVER_H_ */
