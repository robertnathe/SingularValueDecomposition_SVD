#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/QR>

namespace mmio = std; // For clarity in Matrix Market IO namespace usage

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace MatrixMarketIO {

// Writes a sparse matrix in Matrix Market coordinate format
int writeMatrixMarketMatrix(const MatrixXd& matrix, const std::string& filename = "A_out.dat") {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return 1;
    }

    // Count non-zero entries
    size_t nonZeroCount = 0;
    for (int r = 0; r < matrix.rows(); ++r)
        for (int c = 0; c < matrix.cols(); ++c)
            if (matrix(r, c) != 0.0)
                ++nonZeroCount;

    outfile << "%%MatrixMarket matrix coordinate real general\n";
    outfile << matrix.rows() << " " << matrix.cols() << " " << nonZeroCount << "\n";
    outfile << std::fixed << std::setprecision(15);

    for (int r = 0; r < matrix.rows(); ++r) {
        for (int c = 0; c < matrix.cols(); ++c) {
            double val = matrix(r, c);
            if (val != 0.0) {
                outfile << (r + 1) << " " << (c + 1) << " " << val << "\n";
            }
        }
    }
    return 0;
}

// Writes a vector in Matrix Market coordinate format (as a column vector)
int writeMatrixMarketVector(const VectorXd& vector, const std::string& filename = "X_out.dat") {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return 1;
    }

    outfile << "%%MatrixMarket matrix coordinate real general\n";
    outfile << vector.size() << " 1 " << vector.size() << "\n";
    outfile << std::fixed << std::setprecision(15);

    for (int i = 0; i < vector.size(); ++i) {
        outfile << (i + 1) << " 1 " << vector(i) << "\n";
    }
    return 0;
}

// Reads a sparse matrix from a Matrix Market coordinate format file
int readMatrixMarketMatrix(MatrixXd& matrix, const std::string& filename = "A.dat") {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening matrix file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    // Skip comments
    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') break;
    }

    std::istringstream header(line);
    int numRows, numCols, numEntries;
    if (!(header >> numRows >> numCols >> numEntries)) {
        std::cerr << "Invalid header in matrix file: " << filename << std::endl;
        return -1;
    }

    matrix = MatrixXd::Zero(numRows, numCols);

    int row, col;
    double val;
    for (int i = 0; i < numEntries; ++i) {
        if (!(infile >> row >> col >> val)) {
            std::cerr << "Error reading matrix entry #" << i + 1 << std::endl;
            return -1;
        }
        matrix(row - 1, col - 1) = val; // Adjust for 1-based indexing
    }
    return 0;
}

// Reads a vector from a Matrix Market coordinate format file (accepts row or column vector)
int readMatrixMarketVector(VectorXd& vector, const std::string& filename = "B.dat") {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening vector file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    // Skip comments
    while (std::getline(infile, line)) {
        if (!line.empty() && line[0] != '%') break;
    }

    std::istringstream header(line);
    int numRows, numCols, numEntries;
    if (!(header >> numRows >> numCols >> numEntries)) {
        std::cerr << "Invalid header in vector file: " << filename << std::endl;
        return -1;
    }

    if (numCols != 1 && numRows != 1) {
        std::cerr << "Error: file does not represent a vector (must be single row or column)." << std::endl;
        return -1;
    }

    int vectorSize = (numCols == 1) ? numRows : numCols;
    vector = VectorXd::Zero(vectorSize);

    int row, col;
    double val;
    for (int i = 0; i < numEntries; ++i) {
        if (!(infile >> row >> col >> val)) {
            std::cerr << "Error reading vector entry #" << i + 1 << std::endl;
            return -1;
        }
        int idx = (numCols == 1) ? row - 1 : col - 1;
        vector(idx) = val;
    }
    return 0;
}

} // namespace MatrixMarketIO

// Utility: Print elapsed time between two time points
void printExecutionTime(const std::chrono::time_point<std::chrono::system_clock>& start,
                        const std::chrono::time_point<std::chrono::system_clock>& end) {
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
}

// Utility: Print vector with label
void printVector(const std::string& label, const VectorXd& vector) {
    std::cout << label << ":\n[";
    std::cout << std::fixed << std::setprecision(5);
    for (int i = 0; i < vector.size(); ++i) {
        std::cout << vector(i);
        if (i != vector.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
}

// Utility: Print matrix with label
void printMatrix(const std::string& label, const MatrixXd& matrix) {
    std::cout << label << ":\n";
    std::cout << std::fixed << std::setprecision(5);
    for (int r = 0; r < matrix.rows(); ++r) {
        for (int c = 0; c < matrix.cols(); ++c) {
            std::cout << std::setw(12) << matrix(r, c);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    MatrixXd A;
    VectorXd B;

    // Read input matrix and vector
    if (MatrixMarketIO::readMatrixMarketMatrix(A, "A.dat") != 0) {
        std::cerr << "Failed to read matrix A." << std::endl;
        return 1;
    }
    if (MatrixMarketIO::readMatrixMarketVector(B, "B.dat") != 0) {
        std::cerr << "Failed to read vector B." << std::endl;
        return 1;
    }

    if (A.rows() != B.size()) {
        std::cerr << "Dimension mismatch: Matrix A rows (" << A.rows() 
                  << ") != Vector B size (" << B.size() << ")." << std::endl;
        return 1;
    }

    // Compute SVD and solve Ax = B
    auto start = std::chrono::system_clock::now();
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd X = svd.solve(B);
    auto end = std::chrono::system_clock::now();

    // Output results
    printVector("Solution vector X", X);
    printVector("Singular values", svd.singularValues());
    printExecutionTime(start, end);

    // Optionally, write solution vector to file
    if (MatrixMarketIO::writeMatrixMarketVector(X, "X_out.dat") != 0) {
        std::cerr << "Warning: Failed to write solution vector to file." << std::endl;
    }

    return 0;
}
