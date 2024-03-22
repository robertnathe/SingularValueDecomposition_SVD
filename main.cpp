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

int Write1DArrayToMatrixMarketFile(const double B[], int num_rows, int num_cols); 
int Write2DArrayToMatrixMarketFile(const double array_A[], int num_rows, int num_cols);
void print_1DFormatMatrix(const double data[], int num_rows, int num_cols); 
void print_1DFormatVector(const double data[], int num_cols);
int ReadMatrixMarketFile(matrix<double> & A, int& num_rows, int& num_cols); // For matrices 
int ReadVectorMarketFile(vector<double>& B, int& num_cols);
int WriteMatrixMarketFile(const matrix<double>& A, unsigned int num_rows, unsigned int num_cols);
int WriteVectorMarketFile(const std::vector<double>& X, unsigned int num_rows, unsigned int num_cols);
void PrintVector(const std::vector<double>& vector);
void PrintVector2D(const matrix<double>& A);
void print_matrix (double *X, int num_rows, int num_cols);
void Zero_Matrix(double* X, int num_rows, int num_cols);
void Zero_Vector(double* X, int num_cols);
int print_SVD_results(int num_rows, int num_cols, int num_rows_B, int num_cols_B,
                 const double array_U[], const double D[], const double array_V[], const vector<double>& X);
int Parameters_Solve (matrix<double> & AMatrix, int num_rows, int num_cols, double array_U[], double D[], double *V, double B[], vector<double>& X);
int Singular_Value_Decomposition(matrix<double> & AMatrix, double* A, int num_rows, int num_cols, double* U, double* singular_values, double* V, double* dummy_array);
void Singular_Value_Decomposition_Solve(double* U, double* D, double* V, double tolerance, int num_rows, int num_cols, double *B, vector<double>& X);
void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V, double tolerance, int num_rows, int num_cols, double *A_Pseudo_Inverse);

int Write1DArrayToMatrixMarketFile(const double B[], int num_rows, int num_cols) {  
  //Use C++ streams for safer file handling
  ofstream outfile("B_out.dat");
  if (!outfile.is_open()) {
    cerr << "Error opening file for writing: " << "B_out.dat" << endl;
    return 1;
  }
  
  // Use C++ streams for safer file handling
  //ofstream outfile(filename);
  //if (!outfile.is_open()) {
    //cerr << "Error opening file for writing: " << filename << endl;
    //return 1;
  //}
  
  printf ("B =\n");
  // Write header information (assuming general coordinate pattern)
  outfile << "%%MatrixMarket_Output_vector_B.dat matrix coordinate pattern general\n";
  outfile << num_rows << " 1 " << num_rows << endl; // Adjust for 1D array
  // Write each element with row and column indices (starting from 1)
  for (int i = 0; i < num_rows; i++) {
    outfile << i + 1 << " " << 1 << " " << B[i] << endl;
    printf ("%6.5f    ", B[i]);
  }
  std::cout << std::endl;
  outfile.close();
  return 0;
}

int Write2DArrayToMatrixMarketFile(const double array_A[], int num_rows, int num_cols) {
  // Use C++ streams for safer file handling
  ofstream outfile("A_out.dat");
  if (!outfile.is_open()) {
    cerr << "Error opening file for writing: A_out.dat" << endl;
    return 1;
  }

  // Write header information (assuming general coordinate pattern)
  outfile << "%%MatrixMarket_Output_vector_A.dat matrix coordinate pattern general\n";
  outfile << num_rows << " " << num_cols << " " << num_rows * num_cols << endl;
  printf ("A =\n");
  // Write each element with row and column indices (starting from 1)
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      outfile << i + 1 << " " << j + 1 << " " << array_A[i * num_cols + j] << endl;
      printf ("%6.5f    ", array_A[i * num_cols + j]);
    }
    std::cout << std::endl;
  }

  outfile.close();
  return 0;
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

int ReadMatrixMarketFile(matrix<double>& A, int& num_rows, int& num_cols) {
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

// ReadVectorMarketFile implementation for vectors
int ReadVectorMarketFile(vector<double>& B, int& num_cols) {
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

int WriteMatrixMarketFile(const matrix<double>& A, unsigned int num_rows, unsigned int num_cols) {
 // Open file in writing mode
 unsigned int CountNonZeroEntries {0};
 FILE* fptr = fopen("A_out.txt", "w");
 if (fptr == NULL) {
   std::cerr << "Error opening file for writing matrix A." << std::endl;
   return -1; // Indicate error
 }
 for (unsigned int i = 0; i < num_rows; ++i) {
   for (unsigned int j = 0; j < num_cols; ++j) {
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
 for (unsigned int i = 0; i < num_rows; ++i) {
   for (unsigned int j = 0; j < num_cols; ++j) {
     if (A[i][j] != 0.0) 
     { // Check for non-zero value
       fprintf(fptr, "%u %u %lf\n", i + 1, j + 1, A[i][j]);
     }
   }
 }
 // Close the file
 fclose(fptr);
 return 0; // Indicate success
}

int WriteVectorMarketFile(const std::vector<double>& X, unsigned int num_rows, unsigned int num_cols) {
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
 for (unsigned int i = 0; i < num_rows; ++i) {
   fprintf(fptr, "%u %u %lf\n", 1, i+1, X[i]); // Row index always 1 for a vector
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

void PrintVector2D(const matrix<double>& A) {
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
	    printf (" %8.2f ", *X++);
      printf ("\n");
    }
}

void Zero_Matrix(double* X, int num_rows, int num_cols) {
  // Use std::fill for a more concise and readable approach
  std::fill(X, X + num_rows * num_cols, 0.0);
}

void Zero_Vector(double* X, int num_cols) {
  // Use std::fill for a more concise and readable approach
  std::fill(X, X + num_cols, 0.0);
}

int print_SVD_results(int num_rows, int num_cols, int num_rows_B, int num_cols_B,
                 const double array_U[], const double D[], const double array_V[], const vector<double>& X)
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
  WriteVectorMarketFile(X,num_rows,num_cols);  
  return 0;
}

#include "SVDSolver.cpp"
#include "SVD_SOLVER.h"

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
  int Singular_Value_Decomposition_Input(const matrix<double>& AMatrix, int num_rows, int num_cols,
                     double array_U[], double D[], double array_V[],
                     double dummy_array[], double B[], vector<double>& X);
  // Print the pseudo-inverse in a formatted way
  std::cout << "The pseudo-inverse of A = UDV' is A_Pseudo_Inverse =\n";
  print_matrix((double*)A_Pseudo_Inverse, num_rows, num_cols);
  std::cout << std::endl;  // Add an extra newline for clarity
  // Print other results using the provided function
  print_SVD_results(num_rows, num_cols, num_rows, num_cols, array_U, &D[0], array_V, X);
  return 0;
}

int main ()
{	
  // Read A matrix and vector B
  matrix<double> AMatrix;
  vector<double> B;
  int num_rows, num_cols;
  ReadMatrixMarketFile(AMatrix, num_rows, num_cols);
  ReadVectorMarketFile(B, num_cols);
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
