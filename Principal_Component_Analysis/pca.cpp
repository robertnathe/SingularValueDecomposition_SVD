// pca.cpp - Principal Component Analysis using Eigen's SVD
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <thread>          // <-- Added for hardware_concurrency
#include <charconv>        // <-- Added for std::from_chars (if supported)

#ifdef _OPENMP
#include <omp.h>
#endif

using real_t = double;
using MatrixX = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;
using VectorX = Eigen::Vector<real_t, Eigen::Dynamic>;
using BDCSVD = Eigen::BDCSVD<MatrixX>;

class ScopeTimer {
public:
    explicit ScopeTimer(std::string_view name)
        : m_name(name), m_start(std::chrono::steady_clock::now()) {}
    ~ScopeTimer() noexcept {
        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
        std::cout << "[Timer] " << m_name << " took: "
                  << static_cast<double>(duration.count()) / 1000.0 << " ms.\n";
    }
private:
    const std::string_view m_name;
    const std::chrono::steady_clock::time_point m_start;
};

namespace MatrixMarketIO {

void skipHeaderLines(std::istream& file, std::string& line) {
    while (std::getline(file, line)) {
        if (line.empty() || line.front() == '%') continue;
        break;
    }
}

// Fallback parser using stringstream (more compatible than std::from_chars)
class SimpleParser {
public:
    explicit SimpleParser(std::istream& stream) : m_stream(stream) {}

    template<typename T>
    bool next(T& value) {
        m_stream >> value;
        return !m_stream.fail();
    }

private:
    std::istream& m_stream;
};

MatrixX readMatrixMarketMatrix(std::string_view filename) {
    std::ifstream file(filename.data());
    if (!file) throw std::runtime_error("Cannot open file: " + std::string(filename));

    std::string line;
    if (!std::getline(file, line) || line.find("%%MatrixMarket") == std::string::npos)
        throw std::runtime_error("Invalid MatrixMarket header in " + std::string(filename));

    skipHeaderLines(file, line);

    Eigen::Index rows, cols, nnz;
    {
        std::istringstream dims(line);
        dims >> rows >> cols >> nnz;
        if (dims.fail()) throw std::runtime_error("Failed to read matrix dimensions");
    }

    MatrixX mat = MatrixX::Zero(rows, cols);

    // Use simple >> parser for maximum compatibility
    SimpleParser parser(file);
    for (Eigen::Index i = 0; i < nnz; ++i) {
        Eigen::Index r, c;
        real_t val;
        if (!parser.next(r) || !parser.next(c) || !parser.next(val))
            throw std::runtime_error("Parse error in matrix data at entry " + std::to_string(i + 1));
        if (r < 1 || r > rows || c < 1 || c > cols)
            throw std::runtime_error("Index out of bounds");
        mat(r - 1, c - 1) = val;
    }
    return mat;
}

int writeMatrixMarketMatrix(const char* filename, const MatrixX& mat) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << '\n';
        return -1;
    }
    // Count actual non-zeros to be accurate
    Eigen::Index non_zeros = 0;
    for (Eigen::Index j = 0; j < mat.cols(); ++j)
        for (Eigen::Index i = 0; i < mat.rows(); ++i)
            if (std::abs(mat(i, j)) > 1e-15) ++non_zeros;

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << mat.rows() << " " << mat.cols() << " " << non_zeros << "\n";
    file << std::fixed << std::setprecision(15);

    for (Eigen::Index j = 0; j < mat.cols(); ++j) {
        for (Eigen::Index i = 0; i < mat.rows(); ++i) {
            real_t val = mat(i, j);
            if (std::abs(val) > 1e-15) {
                file << (i + 1) << " " << (j + 1) << " " << val << "\n";
            }
        }
    }
    return 0;
}

} // namespace MatrixMarketIO

template<typename Derived>
void printMatrix(const Eigen::MatrixBase<Derived>& m, std::string_view name) {
    Eigen::IOFormat fmt(Eigen::FullPrecision, 0, ", ", "\n", "[", "]", "[", "]");
    std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    std::cout << m.format(fmt) << "\n\n";
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Thread setup
#ifdef _OPENMP
    int threads = omp_get_max_threads();
    Eigen::setNbThreads(threads);
    std::cout << "Using " << threads << " OpenMP threads for Eigen.\n";
#else
    unsigned int threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;  // fallback
    Eigen::setNbThreads(static_cast<int>(threads));
    std::cout << "Using " << threads << " threads for Eigen (std::thread).\n";
#endif

    try {
        MatrixX A;
        {
            ScopeTimer timer("Reading A.dat");
            A = MatrixMarketIO::readMatrixMarketMatrix("A.dat");
        }

        std::cout << "Data matrix A loaded: " << A.rows() << " samples x "
                  << A.cols() << " features\n\n";

        // Center the data
        VectorX colMeans = A.colwise().mean();
        MatrixX A_centered = A.rowwise() - colMeans.transpose();

        // SVD
        BDCSVD svd;
        MatrixX scores;
        MatrixX loadings;
        VectorX singularValues;

        {
            ScopeTimer timer("SVD computation");
            svd.compute(A_centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
            loadings = svd.matrixV();
            singularValues = svd.singularValues();
            scores = A_centered * loadings;
        }

        // Explained variance
        VectorX variance = singularValues.array().square();
        real_t total_variance = variance.sum();
        VectorX explained_variance_ratio = variance / total_variance;

        // Manual cumulative sum (Eigen has no cumsum)
        VectorX cumulative_variance(explained_variance_ratio.size());
        if (explained_variance_ratio.size() > 0) {  // Fixed: use .size() > 0 instead of .empty()
            cumulative_variance(0) = explained_variance_ratio(0);
            for (Eigen::Index i = 1; i < explained_variance_ratio.size(); ++i) {
                cumulative_variance(i) = cumulative_variance(i - 1) + explained_variance_ratio(i);
            }
        }

        printMatrix(scores, "Principal Component Scores (T)");
        printMatrix(loadings, "Loadings (Principal Directions V)");
        std::cout << "Singular values: " << singularValues.transpose() << "\n\n";
        std::cout << "Explained variance ratio: " << explained_variance_ratio.transpose() << "\n";
        std::cout << "Cumulative explained variance: " << cumulative_variance.transpose() << "\n\n";
        std::cout << "Matrix rank: " << svd.rank() << "\n";

        // Write scores
        if (MatrixMarketIO::writeMatrixMarketMatrix("PC_scores.dat", scores) != 0) {
            std::cerr << "WARNING: Failed to write PC_scores.dat\n";
        } else {
            std::cout << "Principal component scores written to PC_scores.dat\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
