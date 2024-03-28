#include <string.h>              // required for memcpy()
#include <float.h>               // required for DBL_EPSILON
#include <math.h>                // required for fabs(), sqrt();
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
using namespace std;
template<typename T>
using matrix = std::vector<std::vector<T>>;

void Zero_Matrix(double* X, int num_rows, int num_cols);
void Zero_Vector(double* X, int num_cols);
void print_1DFormatMatrix(const double data[], int num_rows, int num_cols); 
void print_1DFormatVector(const double data[], int num_cols);
int Write1DArrayToMatrixMarketFile(const double B[], int num_rows); 
int Write2DArrayToMatrixMarketFile(const double array_A[], int num_rows, int num_cols);
int ReadMatrixMarketMatrix(matrix<double> & A, int& num_rows, int& num_cols); // For matrices 
int ReadMatrixMarketVector(vector<double>& B, int& num_cols);
int WriteMatrixMarketMatrix(const matrix<double>& A, int num_rows, int num_cols);
int WriteVectorMarketFile(const std::vector<double>& X, int num_rows, int num_cols);
void PrintVector(const std::vector<double>& vector);
void PrintMatrix(const matrix<double>& A);
void print_matrix (double *X, int num_rows, int num_cols);
int print_SVD_results(int num_rows, int num_cols, const double array_U[], const double D[], const double array_V[], const vector<double>& X);
int Singular_Value_Decomposition_Input(matrix<double> & AMatrix,  int num_rows, int num_cols, double array_U[], double D[], double array_V[], double dummy_array[], double B[], vector<double>& X);
void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,
                double tolerance, int num_rows, int num_cols, double *B, vector<double>& x);
void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V, double tolerance, int num_rows, int num_cols, double *A_Pseudo_Inverse);
#define MAX_ITERATION_COUNT 30   // Maximum number of iterations
//                        Internally Defined Routines
static void Householders_Reduction_to_Bidiagonal_Form(double* A, int num_rows,
    int num_cols, double* U, double* V, double* diagonal, double* superdiagonal );
static int  Givens_Reduction_to_Diagonal_Form( int num_rows, int num_cols,
           double* U, double* V, double* diagonal, double* superdiagonal );
static void Sort_by_Decreasing_Singular_Values(int num_rows, int num_cols,
                                double* singular_value, double* U, double* V);                
int Singular_Value_Decomposition(matrix<double> & AMatrix, double* A, int num_rows, int num_cols, double* U, double* singular_values, double* V, double* dummy_array);

void Zero_Matrix(double* X, int num_rows, int num_cols) {
  // Use std::fill for a more concise and readable approach
  std::fill(X, X + num_rows * num_cols, 0.0);
}

void Zero_Vector(double* X, int num_cols) {
  // Use std::fill for a more concise and readable approach
  std::fill(X, X + num_cols, 0.0);
}

void print_1DFormatMatrix(const double data[], int num_rows, int num_cols) {
  // Use "data" to indicate the content of the array
  for (int row = 0; row < num_rows; ++row) {
    for (int col = 0; col < num_cols; ++col) {
      // Access element using row-major order indexing
      const double element = data[row * num_cols + col];
      std::cout << std::fixed << std::setprecision(4) << std::setw(6)
                << element << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_1DFormatVector(const double data[], int num_elements) {
  // Use "data" to indicate the content of the array
  for (int element_index = 0; element_index < num_elements; element_index++) {
    std::cout << std::fixed << std::setprecision(4) << std::setw(6)
              << data[element_index] << " ";
  }
  std::cout << std::endl << std::endl;
}

int Write1DArrayToMatrixMarketFile(const double B[], int num_rows) {  
  // Use C++ streams for safer file handling
  ofstream outfile("B_out.dat");
  if (!outfile.is_open()) {
    cerr << "Error opening file for writing: " << "B_out.dat" << endl;
    return 1;
  }
  printf ("B =\n");
  // Write header information (assuming general coordinate pattern)
  outfile << "%%MatrixMarket_Output_vector_B.dat matrix coordinate pattern general\n";
  outfile << num_rows << " 1 " << num_rows << endl; // Adjust for 1D array
  // Write each element with row and column indices (starting from 1)
  for (int i = 0; i < num_rows; i++) {
    outfile << i + 1 << " " << 1 << " " << B[i] << endl;
    printf ("%6.5lf    ", B[i]);
  }
  std::cout << std::endl;
  outfile.close();
  return 0;
}

int Write2DArrayToMatrixMarketFile(const double array_A[], int num_rows, int num_cols) {
  // Use C++ streams for safer file handling
  ofstream outfile("A_out.dat");
  if (!outfile.is_open()) {
    cerr << "Error opening file for writing: " << "B_out.dat" << endl;
    return 1;
  }
  printf ("A =\n");
  // Write header information (assuming general coordinate pattern)
  outfile << "%%MatrixMarket_Output_vector_A.dat matrix coordinate pattern general\n";
  outfile << num_rows << " " << num_cols << " " << num_rows * num_cols << endl;
  // Write each element with row and column indices (starting from 1)
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      outfile << i + 1 << " " << j + 1 << " " << array_A[i * num_cols + j] << endl;
      printf ("%6.5lf    ", array_A[i * num_cols + j]);
    }
    std::cout << std::endl;
  }
  outfile.close();
  return 0;
}

int ReadMatrixMarketMatrix(matrix<double>& A, int& num_rows, int& num_cols) {
 int number_of_entries_A {0};
 int i_index {0}, j_index {0};
 double elem {0.0};
 
 FILE* myFile = fopen("A.dat", "r");
 if (myFile == NULL) {
   std::cerr << "Error Reading File" << endl;
   exit(0);
 }
 // Skip header comments
 fscanf(myFile, "%*[^\n]\n"); // Read and discard header line
 // Read dimensions
 if (fscanf(myFile, "%d %d %d\n", &num_rows, &num_cols, &number_of_entries_A) != 3) {
   std::cerr << "Error reading matrix dimensions from A.dat" << endl;
   fclose(myFile);
   return -1;
 }
 // Resize A to accommodate num_rows and num_cols
 A.resize(num_rows);
 for (int i = 0; i < num_rows; ++i) {
   A[i].resize(num_cols);
 }
 // Read non-zero elements by row and column indices
 for (int i = 0; i < number_of_entries_A; ++i) {
   if (fscanf(myFile, "%d %d %lf\n", &i_index, &j_index, &elem) != 3) {
     std::cerr << "Error reading matrix entries from A.dat" << endl;
     fclose(myFile);
     return -1;
   }
   i_index--; // Adjust for zero-based indexing
   j_index--;
   A[i_index][j_index] = elem;
 }
 fclose(myFile);
 return 0;
}

// ReadMatrixMarketVector implementation for vectors
int ReadMatrixMarketVector(vector<double>& B, int& num_cols) {
    (void) num_cols; 
   FILE *myFile2;
   myFile2 = fopen ("B.dat", "r");
   int dim_B_Array[3];
   int i_index {0}, j_index {0};
   double value {0.0};
   while (myFile2 == NULL)
   {
    std::cout << "Error Reading File" << endl;
     exit (0);
   } 
   fscanf (myFile2, "%*s %*s %*s %*s %*s");
   for (int i = 0; i < 3; i++)
   {
     fscanf (myFile2, "%d,", &dim_B_Array[i]);
   }
   for (int i = 0; i < dim_B_Array[1]; i++)
     B.push_back(0.0);
   for (int i = 0; i < dim_B_Array[1]; i++)
   {
     fscanf (myFile2, "%d,", &i_index);
     i_index--;
     fscanf (myFile2, "%d,", &j_index);
     j_index--;
     fscanf (myFile2, "%lf,", &value);
     if (value != 0.0) 
     {
       B[i] = value;
     }
   }
   fclose (myFile2);
 return 0;
}

int WriteMatrixMarketMatrix(const matrix<double>& A, int num_rows, int num_cols) {
 // Open file in writing mode
 int CountNonZeroEntries {0};
 FILE* fptr = fopen("A_out.txt", "w");
 if (fptr == NULL) {
   std::cerr << "Error opening file for writing matrix A." << std::endl;
   return -1; // Indicate error
 }
 for (int i = 0; i < num_rows; ++i) {
   for (int j = 0; j < num_cols; ++j) {
    if (A[i][j] != 0.0)
    {
      CountNonZeroEntries++;
    }
   }
 }
 // Write Matrix Market header information for a sparse matrix
 fprintf(fptr, "%%MatrixMarket_Input_matrix_A.dat matrix coordinate pattern general\n");
// fprintf(fptr, "%d %d %d\n", num_rows, num_cols, num_rows*num_cols); // Count all entries
 fprintf(fptr, "%d %d %d\n", num_rows, num_cols, CountNonZeroEntries); // Count non-zero entries
 // Write only non-zero elements of A
 // I changed this section of code.
 for (int i = 0; i < num_rows; ++i) {
   for (int j = 0; j < num_cols; ++j) {
     if (A[i][j] != 0.0) 
     { // Check for non-zero value
       fprintf(fptr, "%d %d %lf\n", i + 1, j + 1, A[i][j]);
     }
   }
 }
 // Close the file
 fclose(fptr);
 return 0; // Indicate success
}

int WriteMatrixMarketVector(const std::vector<double>& X, int num_rows, int num_cols) {
 // Open file in writing mode
 FILE* fptr = fopen("X.dat", "w");
 if (fptr == NULL) {
   std::cerr << "Error opening file for writing X vector." << std::endl;
   return -1; // Indicate error
 }
 // Write Matrix Market header information
 fprintf(fptr, "%%MatrixMarket_Input_matrix_X.dat matrix coordinate pattern general\n");
 fprintf(fptr, "%d %d %d\n", 1, num_cols, num_cols); // All entries are assumed non-zero
 std::cout << "%%MatrixMarket_Input_matrix_X.dat matrix coordinate pattern general\n";
 // Write each element of X
 for (int i = 0; i < num_rows; ++i) {
   fprintf(fptr, "%d %d %lf\n", 1, i+1, X[i]); // Row index always 1 for a vector
   std::cout << 1 << "   " << i+1 << "   "<< X[i] << std::endl;
 }
 // Close the file
 fclose(fptr);
 return 0; // Indicate success
}

void PrintVector(const std::vector<double>& vector) {
 std::cout << "Displaying the vector: " << endl;
 std::cout << std::setprecision(5);
 for (const double& value : vector) {
   std::cout << value << " ";
 }
 std::cout << std::endl;
}

void PrintMatrix(const matrix<double>& A) {
 std::cout << "Displaying the 2D vector:" << endl;
 for (const auto& row : A) {
   for (const double& value : row) {
     std::cout << value << " ";
   }
   std::cout << std::endl;
 }
}

void print_matrix (double *X, int num_rows, int num_cols)
{
  int i, j;
  for (i = 0; i < num_rows; i++)
    {
      for (j = 0; j < num_cols; j++)
	    printf (" %8.2lf ", *X++);
      printf ("\n");
    }
}

int print_SVD_results(int num_rows, int num_cols, const double array_U[], const double D[], const double array_V[], const vector<double>& X)
{
  printf ("******************** Solve Ax = B ********************\n\n");
  // Print U matrix with proper formatting
  std::cout << "U =\n";
  print_1DFormatMatrix(&array_U[0], num_rows, num_cols); 
  // Print D vector with proper formatting
  std::cout << "D =\n";
  print_1DFormatVector(&D[0], num_cols);
  // Print V matrix with proper formatting
  std::cout << "V =\n";
  print_1DFormatMatrix(&array_V[0], num_rows, num_cols); 
  std::cout << "The solution vector X is the following: " << std::endl << std::endl;
  WriteMatrixMarketVector(X,num_rows,num_cols);  
  return 0;
}

int Singular_Value_Decomposition_Input (matrix<double> & AMatrix,  int num_rows, int num_cols, double array_U[], double D[], double array_V[], double dummy_array[], double B[], vector<double>& X)
{
  int i, j;
  int num_cols_B {0};
  double A[num_rows][num_cols]  = {0.0};  
  double U[num_rows][num_cols] = {0.0};
  double V[num_rows][num_cols]  = {0.0};
  B[num_cols_B] = {0};
  for (i = 0; i < num_rows; i++)
  {
      for (j = 0; j < num_cols; j++)
	{
	  A[i][j] = AMatrix[i][j];
	}
  }
  // Perform SVD
  double tolerance = 0.000001;
  std::cout << "Singular Value Decomposition for Square Matrices" << std::endl;
  Singular_Value_Decomposition(AMatrix, &A[0][0], num_rows, num_cols, &U[0][0], &D[0], &V[0][0], &dummy_array[0]);
  // Solve for X using the SVD
  Singular_Value_Decomposition_Solve(&U[0][0], &D[0], &V[0][0],  tolerance, num_rows, num_cols, &B[0], X);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      array_U[i * num_rows + j] = U[i][j];
	  array_V[i * num_rows + j] = V[i][j];
    }
  }
  // Declare the two-dimensional array
  double A_Pseudo_Inverse[num_rows][num_cols];
  // Initialize the array elements (optional)
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      A_Pseudo_Inverse[i][j] = 0.0; // Example initialization
    }
  }
  Singular_Value_Decomposition_Inverse(&A[0][0], &D[0], &V[0][0], tolerance, num_rows, num_cols, &A_Pseudo_Inverse[0][0]);
  // Print the pseudo-inverse in a formatted way
  std::cout << "The pseudo-inverse of A = UDV' is A_Pseudo_Inverse =\n";
  print_matrix((double*)A_Pseudo_Inverse, num_rows, num_cols);
  std::cout << std::endl;  // Add an extra newline for clarity
  // Print other results using the provided function
  print_SVD_results(num_rows, num_cols, array_U, &D[0], array_V, X);
  return 0;
}
////////////////////////////////////////////////////////////////////////////////
//  void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  //
//              double tolerance, int num_rows, int num_cols, double *B, double* x) //
//                                                                            //
//  Description:                                                              //
//     This routine solves the system of linear equations Ax=B where A =UDV', //
//     is the singular value decomposition of A.  Given UDV'x=B, then         //
//     x = V(1/D)U'B, where 1/D is the pseudo-inverse of D, i.e. if D[i] > 0  //
//     then (1/D)[i] = 1/D[i] and if D[i] = 0, then (1/D)[i] = 0.  Since      //
//     the singular values are subject to round-off error.  A tolerance is    //
//     given so that if D[i] < tolerance, D[i] is treated as if it is 0.      //
//     The default tolerance is D[0] * DBL_EPSILON * num_cols, if the user       //
//     specified tolerance is less than the default tolerance, the default    //
//     tolerance is used.                                                     //
//                                                                            //
//  Arguments:                                                                //
//     double* U                                                              //
//        A matrix with mutually orthonormal columns.                         //
//     double* D                                                              //
//        A diagonal matrix with decreasing non-negative diagonal elements.   //
//        i.e. D[i] > D[j] if i < j and D[i] >= 0 for all i.                  //
//     double* V                                                              //
//        An orthogonal matrix.                                               //
//     double tolerance                                                       //
//        An lower bound for non-zero singular values (provided tolerance >   //
//        num_cols * DBL_EPSILON * D[0]).                                        //
//     int num_rows                                                              //
//        The number of rows of the matrix U and B.                           //
//     int num_cols                                                              //
//        The number of columns of the matrix U.  Also the number of rows and //
//        columns of the matrices D and V.                                    //
//     double* B                                                              //
//        A pointer to a vector dimensioned as num_rows which is the  right-hand //
//        side of the equation Ax = B where A = UDV'.                         //
//     double* x                                                              //
//        A pointer to a vector dimensioned as num_cols, which is the least      //
//        squares solution of the equation Ax = B where A = UDV'.             //
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     #define NB                                                             //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double D[N];                                                           //
//     double B[M];                                                           //
//     double x[N];                                                           //
//     double tolerance;                                                      //
//                                                                            //
//     (your code to initialize the matrices U,D,V,B)                         //
//                                                                            //
//     Singular_Value_Decomposition_Solve((double*) U, D, (double*) V,        //
//                                              tolerance, M, N, B, x, bcols) //
//                                                                            //
//     printf(" The solution of Ax=B is \n");                                 //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //

void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,
                double tolerance, int num_rows, int num_cols, double *B, vector<double>& x)
{
   int i,j,k;
   double *pu, *pv;
   double dum;
   dum = DBL_EPSILON * D[0] * (double) num_cols;
   if (tolerance < dum) tolerance = dum;
   for ( i = 0, pv = V; i < num_cols; i++, pv += num_cols) {
      x[i] = 0.0;
      for (j = 0; j < num_cols; j++)
         if (D[j] > tolerance ) {
            for (k = 0, dum = 0.0, pu = U; k < num_rows; k++, pu += num_cols)
               dum += *(pu + j) * B[k];
            x[i] += dum * *(pv + j) / D[j];
         }
   }
}
////////////////////////////////////////////////////////////////////////////////
//  void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V,//
//                     double tolerance, int num_rows, int num_cols,                //
//                     double *A_Pseudo_Inverse)                              //
//                                                                            //
//  Description:                                                              //
//     This routine calculates the pseudo-inverse of the matrix A = UDV'.     //
//     where U, D, V constitute the singular value decomposition of A.        //
//     Let A_Pseudo_Inverse be the pseudo-inverse then A_Pseudo_Inverse = V(1/D)U', where 1/D is    //
//     the pseudo-inverse of D, i.e. if D[i] > 0 then (1/D)[i] = 1/D[i] and   //
//     if D[i] = 0, then (1/D)[i] = 0.  Because the singular values are       //
//     subject to round-off error.  A tolerance is given so that if           //
//     D[i] < tolerance, D[i] is treated as if it were 0.                     //
//     The default tolerance is D[0] * DBL_EPSILON * num_cols, assuming that the //
//     diagonal matrix of singular values is sorted from largest to smallest, //
//     if the user specified tolerance is less than the default tolerance,    //
//     then the default tolerance is used.                                    //
//                                                                            //
//  Arguments:                                                                //
//     double* U                                                              //
//        A matrix with mutually orthonormal columns.                         //
//     double* D                                                              //
//        A diagonal matrix with decreasing non-negative diagonal elements.   //
//        i.e. D[i] > D[j] if i < j and D[i] >= 0 for all i.                  //
//     double* V                                                              //
//        An orthogonal matrix.                                               //
//     double tolerance                                                       //
//        An lower bound for non-zero singular values (provided tolerance >   //
//        num_cols * DBL_EPSILON * D[0]).                                        //
//     int num_rows                                                              //
//        The number of rows of the matrix U and B.                           //
//     int num_cols                                                              //
//        The number of columns of the matrix U.  Also the number of rows and //
//        columns of the matrices D and V.                                    //
//     double* A_Pseudo_Inverse                                                          //
//        On input, a pointer to the first element of an num_cols x num_rows matrix.//
//        On output, the pseudo-inverse of UDV'.                              //
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double D[N];                                                           //
//     double A_Pseudo_Inverse[N][M];                                                    //
//     double tolerance;                                                      //
//                                                                            //
//     (your code to initialize the matrices U,D,V)                           //
//                                                                            //
//     Singular_Value_Decomposition_Inverse((double*) U, D, (double*) V,      //
//                                        tolerance, M, N, (double*) A_Pseudo_Inverse);  //
//                                                                            //
//     printf(" The pseudo-inverse of A = UDV' is \n");                       //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V, double tolerance, int num_rows, int num_cols, double *A_Pseudo_Inverse)
{
   int i,j,k;
   double *pu, *pv, *pa;
   double dum;
   dum = DBL_EPSILON * D[0] * (double) num_cols;
   if (tolerance < dum) tolerance = dum;
   for ( i = 0, pv = V, pa = A_Pseudo_Inverse; i < num_cols; i++, pv += num_cols)
      for ( j = 0, pu = U; j < num_rows; j++, pa++)
        for (k = 0, *pa = 0.0; k < num_cols; k++, pu++)
           if (D[k] > tolerance) *pa += *(pv + k) * *pu / D[k];
}

////////////////////////////////////////////////////////////////////////////////
// static void Householders_Reduction_to_Bidiagonal_Form(double* A, int num_rows,//
//  int num_cols, double* U, double* V, double* diagonal, double* superdiagonal )//
//                                                                            //
//  Description:                                                              //
//     This routine decomposes an m x n matrix A, with m >= n, into a product //
//     of the three matrices U, B, and V', i.e. A = UBV', where U is an m x n //
//     matrix whose columns are orthogonal, B is a n x n bidiagonal matrix,   //
//     and V is an n x n orthogonal matrix.  V' denotes the transpose of V.   //
//     If m < n, then the procedure may be used for the matrix A'.  The       //
//                                                                            //
//     The matrix U is the product of Householder transformations which       //
//     annihilate the subdiagonal components of A while the matrix V is       //
//     the product of Householder transformations which annihilate the        //
//     components of A to the right of the superdiagonal.                     //
//                                                                            //
//     The Householder transformation which leaves invariant the first k-1    //
//     elements of the k-th column and annihilates the all the elements below //
//     the diagonal element is P = I - (2/u'u)uu', u is an num_rows-dimensional  //
//     vector the first k-1 components of which are zero and the last         //
//     components agree with the current transformed matrix below the diagonal//
//     diagonal, the remaining k-th element is the diagonal element - s, where//
//     s = (+/-)sqrt(sum of squares of the elements below the diagonal), the  //
//     sign is chosen opposite that of the diagonal element.                  //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the pointer to the first element of the matrix            //
//        A[num_rows][num_cols].  The matrix A is unchanged.                        //
//     int num_rows                                                              //
//        The number of rows of the matrix A.                                 //
//     int num_cols                                                              //
//        The number of columns of the matrix A.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix with the same number of rows and    //
//        columns as the matrix A.  On output, the matrix with mutually       //
//        orthogonal columns which is the left-most factor in the bidiagonal  //
//        decomposition of A.                                                 //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix A, i.e. V[num_cols][num_cols].   //
//        On output, the orthogonal matrix whose transpose is the right-most  //
//        factor in the bidiagonal decomposition of A.                        //
//     double* diagonal                                                       //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, num_cols.  On output, the diagonal of the  //
//        bidiagonal matrix.                                                  //
//     double* superdiagonal                                                  //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, num_cols.  On output, the superdiagonal    //
//        of the bidiagonal matrix.                                           //
//                                                                            //
//  Return Values:                                                            //
//     The function is of type void and therefore does not return a value.    //
//     The matrices U, V, and the diagonal and superdiagonal are calculated   //
//     using the addresses passed in the argument list.                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double A[M][N];                                                        //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//     double superdiagonal[N];                                               //
//                                                                            //
//     (your code to initialize the matrix A - Note this routine is not       //
//     (accessible from outside i.e. it is declared static)                   //
//                                                                            //
//     Householders_Reduction_to_Bidiagonal_Form((double*) A, num_rows, num_cols,   //
//                   (double*) U, (double*) V, diagonal, superdiagonal )      //
//                                                                            //
//     free(dummy_array);                                                     //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static void Householders_Reduction_to_Bidiagonal_Form(double* A, int num_rows,
    int num_cols, double* U, double* V, double* diagonal, double* superdiagonal )
{
   int i,j,k,ip1;
   double s, s2, si, scale;
   double *pu, *pui, *pv, *pvi;
   double half_norm_squared;
// Copy A to U
   memcpy(U,A, sizeof(double) * num_rows * num_cols);
//
   diagonal[0] = 0.0;
   s = 0.0;
   scale = 0.0;
   for ( i = 0, pui = U, ip1 = 1; i < num_cols; pui += num_cols, i++, ip1++ ) {
      superdiagonal[i] = scale * s;
//
//                  Perform Householder transform on columns.
//
//       Calculate the normed squared of the i-th column vector starting at
//       row i.
//
      for (j = i, pu = pui, scale = 0.0; j < num_rows; j++, pu += num_cols)
         scale += fabs( *(pu + i) );
      if (scale > 0.0) {
         for (j = i, pu = pui, s2 = 0.0; j < num_rows; j++, pu += num_cols) {
            *(pu + i) /= scale;
            s2 += *(pu + i) * *(pu + i);
         }
//
//
//       Chose sign of s which maximizes the norm
//
         s = ( *(pui + i) < 0.0 ) ? sqrt(s2) : -sqrt(s2);
//
//       Calculate -2/u'u
//
         half_norm_squared = *(pui + i) * s - s2;
//
//       Transform remaining columns by the Householder transform.
//
         *(pui + i) -= s;
         for (j = ip1; j < num_cols; j++) {
            for (k = i, si = 0.0, pu = pui; k < num_rows; k++, pu += num_cols)
               si += *(pu + i) * *(pu + j);
            si /= half_norm_squared;
            for (k = i, pu = pui; k < num_rows; k++, pu += num_cols) {
               *(pu + j) += si * *(pu + i);
            }
         }
      }
      for (j = i, pu = pui; j < num_rows; j++, pu += num_cols) *(pu + i) *= scale;
      diagonal[i] = s * scale;
//
//                  Perform Householder transform on rows.
//
//       Calculate the normed squared of the i-th row vector starting at
//       column i.
//
      s = 0.0;
      scale = 0.0;
      if (i >= num_rows || i == (num_cols - 1) ) continue;
      for (j = ip1; j < num_cols; j++) scale += fabs ( *(pui + j) );
      if ( scale > 0.0 ) {
         for (j = ip1, s2 = 0.0; j < num_cols; j++) {
            *(pui + j) /= scale;
            s2 += *(pui + j) * *(pui + j);
         }
         s = ( *(pui + ip1) < 0.0 ) ? sqrt(s2) : -sqrt(s2);
//
//       Calculate -2/u'u
//
         half_norm_squared = *(pui + ip1) * s - s2;
//
//       Transform the rows by the Householder transform.
//
         *(pui + ip1) -= s;
         for (k = ip1; k < num_cols; k++)
            superdiagonal[k] = *(pui + k) / half_norm_squared;
         if ( i < (num_rows - 1) ) {
            for (j = ip1, pu = pui + num_cols; j < num_rows; j++, pu += num_cols) {
               for (k = ip1, si = 0.0; k < num_cols; k++)
                  si += *(pui + k) * *(pu + k);
               for (k = ip1; k < num_cols; k++) {
                  *(pu + k) += si * superdiagonal[k];
               }
            }
         }
         for (k = ip1; k < num_cols; k++) *(pui + k) *= scale;
      }
   }
// Update V
   pui = U + num_cols * (num_cols - 2);
   pvi = V + num_cols * (num_cols - 1);
   *(pvi + num_cols - 1) = 1.0;
   s = superdiagonal[num_cols - 1];
   pvi -= num_cols;
   for (i = num_cols - 2, ip1 = num_cols - 1; i >= 0; i--, pui -= num_cols,
                                                      pvi -= num_cols, ip1-- ) {
      if ( s != 0.0 ) {
         pv = pvi + num_cols;
         for (j = ip1; j < num_cols; j++, pv += num_cols)
            *(pv + i) = ( *(pui + j) / *(pui + ip1) ) / s;
         for (j = ip1; j < num_cols; j++) {
            si = 0.0;
            for (k = ip1, pv = pvi + num_cols; k < num_cols; k++, pv += num_cols)
               si += *(pui + k) * *(pv + j);
            for (k = ip1, pv = pvi + num_cols; k < num_cols; k++, pv += num_cols)
               *(pv + j) += si * *(pv + i);
         }
      }
      pv = pvi + num_cols;
      for ( j = ip1; j < num_cols; j++, pv += num_cols ) {
         *(pvi + j) = 0.0;
         *(pv + i) = 0.0;
      }
      *(pvi + i) = 1.0;
      s = superdiagonal[i];
   }
// Update U
   pui = U + num_cols * (num_cols - 1);
   for (i = num_cols - 1, ip1 = num_cols; i >= 0; ip1 = i, i--, pui -= num_cols ) {
      s = diagonal[i];
      for ( j = ip1; j < num_cols; j++) *(pui + j) = 0.0;
      if ( s != 0.0 ) {
         for (j = ip1; j < num_cols; j++) {
            si = 0.0;
            pu = pui + num_cols;
            for (k = ip1; k < num_rows; k++, pu += num_cols)
               si += *(pu + i) * *(pu + j);
            si = (si / *(pui + i) ) / s;
            for (k = i, pu = pui; k < num_rows; k++, pu += num_cols)
               *(pu + j) += si * *(pu + i);
         }
         for (j = i, pu = pui; j < num_rows; j++, pu += num_cols){
            *(pu + i) /= s;
         }
      }
      else
         for (j = i, pu = pui; j < num_rows; j++, pu += num_cols) *(pu + i) = 0.0;
      *(pui + i) += 1.0;
   }
}
////////////////////////////////////////////////////////////////////////////////
// static int Givens_Reduction_to_Diagonal_Form( int num_rows, int num_cols,        //
//         double* U, double* V, double* diagonal, double* superdiagonal )    //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes a bidiagonal matrix given by the arrays        //
//     diagonal and superdiagonal into a product of three matrices U1, D and  //
//     V1', the matrix U1 premultiplies U and is returned in U, the matrix    //
//     V1 premultiplies V and is returned in V.  The matrix D is a diagonal   //
//     matrix and replaces the array diagonal.                                //
//                                                                            //
//     The method used to annihilate the offdiagonal elements is a variant    //
//     of the QR transformation.  The method consists of applying Givens      //
//     rotations to the right and the left of the current matrix until        //
//     the new off-diagonal elements are chased out of the matrix.            //
//                                                                            //
//     The process is an iterative process which due to roundoff errors may   //
//     not converge within a predefined number of iterations.  (This should   //
//     be unusual.)                                                           //
//                                                                            //
//  Arguments:                                                                //
//     int num_rows                                                              //
//        The number of rows of the matrix U.                                 //
//     int num_cols                                                              //
//        The number of columns of the matrix U.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix already initialized to a matrix     //
//        with mutually orthogonal columns.   On output, the matrix with      //
//        mutually orthogonal columns.                                        //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix U, i.e. V[num_cols][num_cols].   //
//        The matrix V is assumed to be initialized to an orthogonal matrix.  //
//        On output, V is an orthogonal matrix.                               //
//     double* diagonal                                                       //
//        On input, a pointer to an array of dimension num_cols which initially  //
//        contains the diagonal of the bidiagonal matrix.  On output, the     //
//        it contains the diagonal of the diagonal matrix.                    //
//     double* superdiagonal                                                  //
//        On input, a pointer to an array of dimension num_cols which initially  //
//        the first component is zero and the successive components form the  //
//        superdiagonal of the bidiagonal matrix.                             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The procedure failed to terminate within                  //
//                  MAX_ITERATION_COUNT iterations.                           //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//     double superdiagonal[N];                                               //
//     int err;                                                               //
//                                                                            //
//     (your code to initialize the matrices U, V, diagonal, and )            //
//     ( superdiagonal.  - Note this routine is not accessible from outside)  //
//     ( i.e. it is declared static.)                                         //
//                                                                            //
//     err = Givens_Reduction_to_Diagonal_Form( M,N,(double*)U,(double*)V,    //
//                                                 diagonal, superdiagonal ); //
//     if ( err < 0 ) printf("Failed to converge\n");                         //
//     else { ... }                                                           //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static int Givens_Reduction_to_Diagonal_Form( int num_rows, int num_cols,
           double* U, double* V, double* diagonal, double* superdiagonal )
{
   double epsilon;
   double c, s;
   double f,g,h;
   double x,y,z;
   double *pu, *pv;
   int i,j,k,m;
   int rotation_test;
   int iteration_count;

   for (i = 0, x = 0.0; i < num_cols; i++) {
      y = fabs(diagonal[i]) + fabs(superdiagonal[i]);
      if ( x < y ) x = y;
   }
   epsilon = x * DBL_EPSILON;
   for (k = num_cols - 1; k >= 0; k--) {
      iteration_count = 0;
      while(1) {
         rotation_test = 1;
         for (m = k; m >= 0; m--) {
            if (fabs(superdiagonal[m]) <= epsilon) {rotation_test = 0; break;}
            if (fabs(diagonal[m-1]) <= epsilon) break;
         }
         if (rotation_test) {
            c = 0.0;
            s = 1.0;
            for (i = m; i <= k; i++) {
               f = s * superdiagonal[i];
               superdiagonal[i] *= c;
               if (fabs(f) <= epsilon) break;
               g = diagonal[i];
               h = sqrt(f*f + g*g);
               diagonal[i] = h;
               c = g / h;
               s = -f / h;
               for (j = 0, pu = U; j < num_rows; j++, pu += num_cols) {
                  y = *(pu + m - 1);
                  z = *(pu + i);
                  *(pu + m - 1 ) = y * c + z * s;
                  *(pu + i) = -y * s + z * c;
               }
            }
         }
         z = diagonal[k];
         if (m == k ) {
            if ( z < 0.0 ) {
               diagonal[k] = -z;
               for ( j = 0, pv = V; j < num_cols; j++, pv += num_cols)
                  *(pv + k) = - *(pv + k);
            }
            break;
         }
         else {
            if ( iteration_count >= MAX_ITERATION_COUNT ) return -1;
            iteration_count++;
            x = diagonal[m];
            y = diagonal[k-1];
            g = superdiagonal[k-1];
            h = superdiagonal[k];
            f = ( (y - z) * ( y + z ) + (g - h) * (g + h) )/(2.0 * h * y);
            g = sqrt( f * f + 1.0 );
            if ( f < 0.0 ) g = -g;
            f = ( (x - z) * (x + z) + h * (y / (f + g) - h) ) / x;
// Next QR Transformtion
            c = 1.0;
            s = 1.0;
            for (i = m + 1; i <= k; i++) {
               g = superdiagonal[i];
               y = diagonal[i];
               h = s * g;
               g *= c;
               z = sqrt( f * f + h * h );
               superdiagonal[i-1] = z;
               c = f / z;
               s = h / z;
               f =  x * c + g * s;
               g = -x * s + g * c;
               h = y * s;
               y *= c;
               for (j = 0, pv = V; j < num_cols; j++, pv += num_cols) {
                  x = *(pv + i - 1);
                  z = *(pv + i);
                  *(pv + i - 1) = x * c + z * s;
                  *(pv + i) = -x * s + z * c;
               }
               z = sqrt( f * f + h * h );
               diagonal[i - 1] = z;
               if (z != 0.0) {
                  c = f / z;
                  s = h / z;
               }
               f = c * g + s * y;
               x = -s * g + c * y;
               for (j = 0, pu = U; j < num_rows; j++, pu += num_cols) {
                  y = *(pu + i - 1);
                  z = *(pu + i);
                  *(pu + i - 1) = c * y + s * z;
                  *(pu + i) = -s * y + c * z;
               }
            }
            superdiagonal[m] = 0.0;
            superdiagonal[k] = f;
            diagonal[k] = x;
         }
      }
   }
   return 0;
}
////////////////////////////////////////////////////////////////////////////////
// static void Sort_by_Decreasing_Singular_Values(int num_rows, int num_cols,       //
//                            double* singular_values, double* U, double* V)  //
//                                                                            //
//  Description:                                                              //
//     This routine sorts the singular values from largest to smallest        //
//     singular value and interchanges the columns of U and the columns of V  //
//     whenever a swap is made.  I.e. if the i-th singular value is swapped   //
//     with the j-th singular value, then the i-th and j-th columns of U are  //
//     interchanged and the i-th and j-th columns of V are interchanged.      //
//                                                                            //
//  Arguments:                                                                //
//     int num_rows                                                              //
//        The number of rows of the matrix U.                                 //
//     int num_cols                                                              //
//        The number of columns of the matrix U.                              //
//     double* singular_values                                                //
//        On input, a pointer to the array of singular values.  On output, the//
//        sorted array of singular values.                                    //
//     double* U                                                              //
//        On input, a pointer to a matrix already initialized to a matrix     //
//        with mutually orthogonal columns.  On output, the matrix with       //
//        mutually orthogonal possibly permuted columns.                      //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix U, i.e. V[num_cols][num_cols].   //
//        The matrix V is assumed to be initialized to an orthogonal matrix.  //
//        On output, V is an orthogonal matrix with possibly permuted columns.//
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//                                                                            //
//     (your code to initialize the matrices U, V, and diagonal. )            //
//     ( - Note this routine is not accessible from outside)                  //
//     ( i.e. it is declared static.)                                         //
//                                                                            //
//     Sort_by_Decreasing_Singular_Values(num_rows, num_cols, singular_values,      //
//                                                 (double*) U, (double*) V); //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static void Sort_by_Decreasing_Singular_Values(int num_rows, int num_cols,
                                double* singular_values, double* U, double* V)
{
   int i,j,max_index;
   double temp;
   double *p1, *p2;

   for (i = 0; i < num_cols - 1; i++) {
      max_index = i;
      for (j = i + 1; j < num_cols; j++)
         if (singular_values[j] > singular_values[max_index] )
            max_index = j;
      if (max_index == i) continue;
      temp = singular_values[i];
      singular_values[i] = singular_values[max_index];
      singular_values[max_index] = temp;
      p1 = U + max_index;
      p2 = U + i;
      for (j = 0; j < num_rows; j++, p1 += num_cols, p2 += num_cols) {
         temp = *p1;
         *p1 = *p2;
         *p2 = temp;
      }
      p1 = V + max_index;
      p2 = V + i;
      for (j = 0; j < num_cols; j++, p1 += num_cols, p2 += num_cols) {
         temp = *p1;
         *p1 = *p2;
         *p2 = temp;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
// File: singular_value_decomposition.c                                       //
// Contents:                                                                  //
//    Singular_Value_Decomposition                                            //
//    Singular_Value_Decomposition_Solve                                      //
//    Singular_Value_Decomposition_Inverse                                    //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  int Singular_Value_Decomposition(double* A, int num_rows, int num_cols,         //
//        double* U, double* singular_values, double* V, double* dummy_array) //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes an m x n matrix A, with m >= n, into a product //
//     of the three matrices U, D, and V', i.e. A = UDV', where U is an m x n //
//     matrix whose columns are orthogonal, D is a n x n diagonal matrix, and //
//     V is an n x n orthogonal matrix.  V' denotes the transpose of V.  If   //
//     m < n, then the procedure may be used for the matrix A'.  The singular //
//     values of A are the diagonal elements of the diagonal matrix D and     //
//     correspond to the positive square roots of the eigenvalues of the      //
//     matrix A'A.                                                            //
//                                                                            //
//     This procedure programmed here is based on the method of Golub and     //
//     Reinsch as given on pages 134 - 151 of the "Handbook for Automatic     //
//     Computation vol II - Linear Algebra" edited by Wilkinson and Reinsch   //
//     and published by Springer-Verlag, 1971.                                //
//                                                                            //
//     The Golub and Reinsch's method for decomposing the matrix A into the   //
//     product U, D, and V' is performed in three stages:                     //
//       Stage 1:  Decompose A into the product of three matrices U1, B, V1'  //
//         A = U1 B V1' where B is a bidiagonal matrix, and U1, and V1 are a  //
//         product of Householder transformations.                            //
//       Stage 2:  Use Given' transformations to reduce the bidiagonal matrix //
//         B into the product of the three matrices U2, D, V2'.  The singular //
//         value decomposition is then UDV'where U = U2 U1 and V' = V1' V2'.  //
//       Stage 3:  Sort the matrix D in decreasing order of the singular      //
//         values and interchange the columns of both U and V to reflect any  //
//         change in the order of the singular values.                        //
//                                                                            //
//     After performing the singular value decomposition for A, call          //
//     Singular_Value_Decomposition to solve the equation Ax = B or call      //
//     Singular_Value_Decomposition_Inverse to calculate the pseudo-inverse   //
//     of A.                                                                  //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the pointer to the first element of the matrix            //
//        A[num_rows][num_cols].  The matrix A is unchanged.                        //
//     int num_rows                                                              //
//        The number of rows of the matrix A.                                 //
//     int num_cols                                                              //
//        The number of columns of the matrix A.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix with the same number of rows and    //
//        columns as the matrix A.  On output, the matrix with mutually       //
//        orthogonal columns which is the left-most factor in the singular    //
//        value decomposition of A.                                           //
//     double* singular_values                                                //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, num_cols.  On output, the singular values  //
//        of the matrix A sorted in decreasing order.  This array corresponds //
//        to the diagonal matrix in the singular value decomposition of A.    //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix A, i.e. V[num_cols][num_cols].   //
//        On output, the orthogonal matrix whose transpose is the right-most  //
//        factor in the singular value decomposition of A.                    //
//     double* dummy_array                                                    //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, num_cols.  This array is used to store     //
//        the super-diagonal elements resulting from the Householder reduction//
//        of the matrix A to bidiagonal form.  And as an input to the Given's //
//        procedure to reduce the bidiagonal form to diagonal form.           //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - During the Given's reduction of the bidiagonal form to    //
//                  diagonal form the procedure failed to terminate within    //
//                  MAX_ITERATION_COUNT iterations.                           //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double A[M][N];                                                        //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double singular_values[N];                                             //
//     double* dummy_array;                                                   //
//                                                                            //
//     (your code to initialize the matrix A)                                 //
//     dummy_array = (double*) malloc(N * sizeof(double));                    //
//     if (dummy_array == NULL) {printf(" No memory available\n"); exit(0); } //
//                                                                            //
//     err = Singular_Value_Decomposition((double*) A, M, N, (double*) U,     //
//                              singular_values, (double*) V, dummy_array);   //
//                                                                            //
//     free(dummy_array);                                                     //
//     if (err < 0) printf(" Failed to converge\n");                          //
//     else { printf(" The singular value decomposition of A is \n");         //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Singular_Value_Decomposition(matrix<double> & AMatrix, double* A, int num_rows, int num_cols, double* U,
                      double* singular_values, double* V, double* dummy_array)
{
   Householders_Reduction_to_Bidiagonal_Form( A, num_rows, num_cols, U, V, singular_values, dummy_array);
   if (Givens_Reduction_to_Diagonal_Form( num_rows, num_cols, U, V, singular_values, dummy_array ) < 0) return -1;
   Sort_by_Decreasing_Singular_Values(num_rows, num_cols, singular_values, U, V);
   return 0;
}

int main ()
{	
  // Read A matrix and vector B
  matrix<double> AMatrix;
  vector<double> B;
  int num_rows, num_cols;
  ReadMatrixMarketMatrix(AMatrix, num_rows, num_cols);
  ReadMatrixMarketVector(B, num_cols);
  // Allocate memory for U and V on the heap for efficiency
  double* array_U = new double[num_rows * num_cols];
  double* array_V = new double[num_rows * num_cols];
  // Copy A matrix into U and V arrays
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      array_U[i * num_rows + j] = AMatrix[i][j];
      array_V[i * num_rows + j] = AMatrix[i][j];
    }
  }
  // Allocate memory for D, dummy_array, and X on the heap
  vector<double> D(num_cols);
  vector<double> dummy_array(num_cols);
  vector<double> X(num_cols);
  // Call Singular_Value_Decomposition_Input function
  Singular_Value_Decomposition_Input(AMatrix, num_rows, num_cols, array_U, &D[0], array_V, &dummy_array[0], &B[0], X);
  // Deallocate memory allocated on the heap
  delete[] array_U;
  delete[] array_V;
  return 0;
}
