// Optimal compilation flags for performance with Eigen:
// g++/clang++: -O3 -DNDEBUG -march=native -fopenmp -std=c++20
// MSVC: /O2 /D NDEBUG /arch:AVX2 /openmp:llvm /std:c++20
// Note: Eigen can leverage OpenMP for parallelization. Ensure it's enabled.

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

// Type aliases for easy configuration and clarity
using real_t = double;
// Use Eigen's default ColMajor storage for performance with BLAS/LAPACK.
using MatrixX = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;
using VectorX = Eigen::Vector<real_t, Eigen::Dynamic>;
// BDCSVD is a high-performance divide-and-conquer SVD for large matrices.
using BDCSVD = Eigen::BDCSVD<MatrixX>;

// A simple RAII timer for measuring execution time in a scope
class ScopeTimer {
public:
    explicit ScopeTimer(std::string_view name) 
        : m_name(name), m_start(std::chrono::steady_clock::now()) {}

    ~ScopeTimer() noexcept {
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        std::cout << m_name << " took: " 
                  << static_cast<double>(duration.count()) / 1000.0 << " ms.\n";
    }

    // Disable copy/move operations for safety
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;
    ScopeTimer(ScopeTimer&&) = delete;
    ScopeTimer& operator=(ScopeTimer&&) = delete;

private:
    const std::string_view m_name;
    const std::chrono::steady_clock::time_point m_start;
};

namespace MatrixMarketIO {
namespace { // Anonymous namespace for internal linkage helpers

// Skips comment and blank lines in MatrixMarket files
void skipHeaderLines(std::istream& file, std::string& line) {
    while (std::getline(file, line)) {
        if (line.empty() || line.front() == '%') {
            continue;
        }
        break; // Found the first non-comment, non-empty line
    }
}

// A high-performance parser for whitespace-separated values from a string_view.
// Utilizes C++20's std::from_chars for maximum speed.
class ViewParser {
public:
    explicit ViewParser(std::string_view sv) noexcept : m_sv(sv) {}

    // Parses the next value of type T from the view.
    // Returns true on success, false on failure or end of view.
    template<typename T>
    [[nodiscard]] bool next(T& value) noexcept {
        skipWhitespace();
        if (m_sv.empty()) {
            return false;
        }
        const auto result = std::from_chars(m_sv.data(), m_sv.data() + m_sv.size(), value);
        if (result.ec != std::errc{}) {
            return false;
        }
        m_sv.remove_prefix(result.ptr - m_sv.data());
        return true;
    }

private:
    void skipWhitespace() noexcept {
        const auto pos = m_sv.find_first_not_of(" \t\n\r");
        if (pos != std::string_view::npos) {
            m_sv.remove_prefix(pos);
        } else {
            m_sv.remove_prefix(m_sv.size()); // All whitespace, make view empty
        }
    }
    std::string_view m_sv;
};

} // anonymous namespace

[[nodiscard]] MatrixX readMatrixMarketMatrix(std::string_view filename) {
    std::ifstream file(filename.data());
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open matrix file: " + std::string(filename));
    }
    file.exceptions(std::ifstream::badbit);

    std::string line;
    if (!std::getline(file, line) || line.find("%%MatrixMarket") == std::string::npos) {
        throw std::runtime_error("Invalid MatrixMarket header in " + std::string(filename));
    }

    skipHeaderLines(file, line);

    std::istringstream dims_stream(line);
    Eigen::Index rows, cols, nonZeros;
    dims_stream >> rows >> cols >> nonZeros;

    if (dims_stream.fail() || rows <= 0 || cols <= 0 || nonZeros < 0) {
        throw std::runtime_error("Invalid matrix dimensions in " + std::string(filename));
    }

    MatrixX mat = MatrixX::Zero(rows, cols);
    
    // Optimization: Read rest of file into stringstream, then parse with from_chars.
    std::stringstream buffer;
    buffer << file.rdbuf();
    ViewParser parser(buffer.view());

    for (Eigen::Index i = 0; i < nonZeros; ++i) {
        Eigen::Index row_idx, col_idx;
        real_t value;
        if (!parser.next(row_idx) || !parser.next(col_idx) || !parser.next(value)) {
             throw std::runtime_error("Error reading matrix data at non-zero entry " + std::to_string(i + 1));
        }
        if (row_idx < 1 || row_idx > rows || col_idx < 1 || col_idx > cols) {
            throw std::runtime_error("Invalid matrix index at non-zero entry " + std::to_string(i + 1));
        }
        mat(row_idx - 1, col_idx - 1) = value;
    }
    return mat;
}

[[nodiscard]] VectorX readMatrixMarketVector(std::string_view filename) {
    std::ifstream file(filename.data());
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vector file: " + std::string(filename));
    }
    file.exceptions(std::ifstream::badbit);

    std::string line;
    if (!std::getline(file, line) || line.find("%%MatrixMarket") == std::string::npos) {
        throw std::runtime_error("Invalid MatrixMarket header in " + std::string(filename));
    }

    skipHeaderLines(file, line);

    std::istringstream dims_stream(line);
    Eigen::Index rows, cols, nonZeros;
    dims_stream >> rows >> cols >> nonZeros;

    if (dims_stream.fail() || rows <= 0 || cols <= 0 || (rows != 1 && cols != 1)) {
        throw std::runtime_error("File is not a vector (must have 1 row or 1 column): " + std::string(filename));
    }
    
    const Eigen::Index size = std::max(rows, cols);
    VectorX vec = VectorX::Zero(size);

    std::stringstream buffer;
    buffer << file.rdbuf();
    ViewParser parser(buffer.view());

    for (Eigen::Index i = 0; i < nonZeros; ++i) {
        Eigen::Index idx1, idx2;
        real_t value;
        if (!parser.next(idx1) || !parser.next(idx2) || !parser.next(value)) {
             throw std::runtime_error("Error reading vector data at non-zero entry " + std::to_string(i + 1));
        }

        if (rows == 1) { // Row vector
             if (idx1 != 1 || idx2 < 1 || idx2 > size) {
                throw std::runtime_error("Invalid row vector index at non-zero entry " + std::to_string(i + 1));
            }
            vec(idx2 - 1) = value;
        } else { // Column vector
            if (idx2 != 1 || idx1 < 1 || idx1 > size) {
                throw std::runtime_error("Invalid column vector index at non-zero entry " + std::to_string(i + 1));
            }
            vec(idx1 - 1) = value;
        }
    }
    return vec;
}

[[nodiscard]] int WriteMatrixMarketVector(const char* filename, const std::vector<double>& vec) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << '\n';
        return -1;
    }
    file << "%%MatrixMarket matrix array real general\n";
    file << "1 " << vec.size() << " "<< vec.size() << '\n';
    file << std::fixed << std::setprecision(15);
    int i = 1;
    for (const auto& val : vec) {
        file << 1 << " " << i << " " << val << '\n';
        i++;
    }
    return 0;
}}

template<typename Derived>
void printVector(const Eigen::DenseBase<Derived>& v, std::string_view name) noexcept {
    constexpr Eigen::Index limit = 10;
    const Eigen::Index size = v.size();
    const Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");

    std::cout << name << " (size " << size << "):\n";
    if (size > 2 * limit) {
        std::cout << v.head(limit).format(HeavyFmt) << "\n...\n" << v.tail(limit).format(HeavyFmt) << "\n\n";
    } else {
        std::cout << v.format(HeavyFmt) << "\n\n";
    }
}

int main() {
    // Fast C++ I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Explicitly configure Eigen to use multiple threads.
    // This is crucial for performance on multi-core systems.
#ifdef _OPENMP
    const int num_threads = omp_get_max_threads();
    Eigen::setNbThreads(num_threads);
    std::cout << "Using " << num_threads << " threads for Eigen (OpenMP).\n";
#else
    const unsigned int num_threads = std::thread::hardware_concurrency();
    Eigen::setNbThreads(static_cast<int>(num_threads));
    std::cout << "Using " << num_threads << " threads for Eigen (std::thread).\n";
#endif
    
    try {
        MatrixX A;
        VectorX b;
        {
            ScopeTimer io_timer("File I/O");
            A = MatrixMarketIO::readMatrixMarketMatrix("A.dat");
            b = MatrixMarketIO::readMatrixMarketVector("B.dat");
        }

        if (A.rows() != b.size()) {
            std::cerr << "\nDimension Mismatch Error:\n"
                      << "Matrix A has " << A.rows() << " rows, but vector b has " << b.size() << " elements.\n"
                      << "For Ax = b, the number of rows in A must equal the size of b.\n";
            return EXIT_FAILURE;
        }

        std::cout << "\nSolving linear system Ax=b:\n"
                  << "  A: " << A.rows() << " x " << A.cols() << "\n"
                  << "  b: " << b.size() << " x 1\n\n";

        VectorX x;
        BDCSVD svd;
        {
            ScopeTimer solve_timer("SVD computation and solve");
            // BDCSVD is robust and accurate, suitable for all matrix types.
            // ComputeThinU and ComputeThinV are sufficient and faster for solving.
            svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            x = svd.solve(b);
        }

        printVector(x, "Solution vector X");
        
        const VectorX& singularValues = svd.singularValues();
        printVector(singularValues, "Singular values");
        
        std::cout << "Matrix rank: " << svd.rank() << "\n";

        // Robustly calculate condition number: ratio of largest to smallest singular value.
        real_t condition_number = std::numeric_limits<real_t>::infinity();
        if (svd.rank() > 0) {
            const real_t max_singular = singularValues(0);
            const real_t min_nonzero_singular = singularValues(svd.rank() - 1);
            
            // Check if smallest non-zero singular value is numerically significant.
            if (min_nonzero_singular > std::numeric_limits<real_t>::epsilon() * max_singular) {
                condition_number = max_singular / min_nonzero_singular;
            }
        }
        std::cout << "Condition number: " << condition_number << "\n\n";
		if (MatrixMarketIO::WriteMatrixMarketVector("X_out.dat", {x.data(), x.data() + x.size()}) != 0) {
		    std::cerr << "WARNING: Failed to write solution to X_out.dat\n";
		}
        std::cout << "Solution written to X_out.dat\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
