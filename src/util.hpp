#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <vector>
#include <complex>

typedef std::complex<double> cplx;
typedef std::vector<cplx> cvec;

#define IM cplx(0.0, 1.0)
#define PI 3.14159265358979323846


/**
 * @brief Class representing a penta-diagonal matrix.
 */
class PentaDiagonalMatrix {
public:
    /**
     * @brief Construct an empty penta-diagonal matrix of given size.
     * @param size Size of the matrix (size x size)
     */
    PentaDiagonalMatrix(int size) : size(size) {
        matrix.resize(5, std::vector<cplx>(size, cplx(0.0, 0.0))); 
    }

    /**
     * @brief Construct a penta-diagonal matrix with specified diagonal values.
     * @param size Size of the matrix (size x size)
     * @param a Value for the sub-sub-diagonal
     * @param b Value for the sub-diagonal
     * @param c Value for the main diagonal 
     * @param d Value for the super-diagonal
     * @param e Value for the super-super-diagonal
     */
    template<typename T>
    PentaDiagonalMatrix(int size, const T& a, const T& b, const T& c, const T& d, const T& e) : size(size) {
        matrix.resize(5, std::vector<cplx>(size, cplx(0.0, 0.0))); 
        std::fill(matrix[0].begin(), matrix[0].end(), cplx(a));
        std::fill(matrix[1].begin(), matrix[1].end(), cplx(b));
        std::fill(matrix[2].begin(), matrix[2].end(), cplx(c));
        std::fill(matrix[3].begin(), matrix[3].end(), cplx(d));
        std::fill(matrix[4].begin(), matrix[4].end(), cplx(e));
    }

    /**
     * @brief Construct a penta-diagonal matrix with specified main diagonal values.
     * @param size Size of the matrix (size x size)
     * @param diagonal_data Vector containing values for the main diagonal
     */
    template<typename T>
    PentaDiagonalMatrix(int size, const std::vector<T>& diagonal_data) : size(size) {
        matrix.resize(5, std::vector<cplx>(size, cplx(0.0, 0.0))); 
        for (int j = 0; j < size; ++j) {
            matrix[2][j] = cplx(diagonal_data[j]);  // main diagonal
        }
    }

    /**
     * @brief Copy constructor
     */
    PentaDiagonalMatrix(const PentaDiagonalMatrix& other) {
        size = other.size;
        matrix = other.matrix;
    }

    /**
     * @brief Multiply the matrix by a scalar.
     */
    template<typename T>
    PentaDiagonalMatrix& operator*=(T scalar) {
        for (int i = 0; i < band_width; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] *= scalar;
            }
        }
        return *this;
    }

    /**
     * @brief Multiply the matrix by a scalar (returns new matrix).
     */
    template<typename T>
    PentaDiagonalMatrix operator*(T scalar) const {
        PentaDiagonalMatrix result(*this);
        for (int i = 0; i < band_width; ++i) {
            for (int j = 0; j < size; ++j) {
                result.matrix[i][j] *= scalar;
            }
        }
        return result;
    }

    /**
     * @brief Multiply the matrix by a vector.
     */
    template<typename T>
    std::vector<cplx> operator*(const std::vector<T>& vec) const {
        std::vector<cplx> ans(size, cplx(0.0, 0.0));
        for (int i = 0; i < size; ++i) {
            for (int j = -2; j <= 2; ++j) {
                int col = i + j;
                if (col >= 0 && col < size) {
                    ans[i] += matrix[j + 2][i] * cplx(vec[col]);
                }
            }
        }
        return ans;
    }

    template<typename T>
    friend PentaDiagonalMatrix operator*(T scalar, const PentaDiagonalMatrix& matrix) {
        return matrix * scalar;
    }

    /**
     * @brief Add another penta-diagonal matrix to this matrix.
     */
    PentaDiagonalMatrix& operator+=(const PentaDiagonalMatrix& other) {
        for (int i = 0; i < band_width; ++i) {
            for (int j = 0; j < size; ++j) {
                this->matrix[i][j] += other.matrix[i][j];
            }
        }
        return *this;
    }

    /**
     * @brief Add another penta-diagonal matrix to this matrix (returns new matrix).
     */
    PentaDiagonalMatrix operator+(const PentaDiagonalMatrix& other) const {
        PentaDiagonalMatrix result(*this);
        for (int i = 0; i < band_width; ++i) {
            for (int j = 0; j < size; ++j) {
                result.matrix[i][j] += other.matrix[i][j];
            }
        }
        return result;
    }

    /**
     * @brief Assignment operator
     */
    PentaDiagonalMatrix& operator=(const PentaDiagonalMatrix& other) {
        this->size = other.size;
        this->matrix = other.matrix;
        return *this;
    }

    /**
     * @brief Clear the matrix (set all elements to zero).
     */
    void clear() {
        for (int i = 0; i < band_width; ++i) {
            std::fill(matrix[i].begin(), matrix[i].end(), cplx(0.0, 0.0));
        }
    }

public:
    int size;                                   // the size of the matrix (size x size)
    std::vector<std::vector<cplx> > matrix;     // 5 x size matrix storing the diagonals
private:
    const int band_width = 5;                   // number of diagonals in penta-diagonal matrix
};


// helper functions for penta-diagonal matrix indexing
inline int ptxid(int i, int j) {
    return (std::min(i, j)) - 1;
}
inline int ptyid(int i, int j) {
    return (j - i + 3) - 1;
}

/**
 * @brief Solve the linear system A * X = B for X, using gauss-elimination tailored for penta-diagonal matrices.
 * 
 * @param A_buffer Temporary buffer for matrix A modifications
 * @param B_buffer Temporary buffer for vector B modifications
 * @param A Coefficient matrix (penta-diagonal)
 * @param B Right-hand side vector
 * @param X Solution vector (output)
 */
inline void solve_linear_system(PentaDiagonalMatrix& A_buffer, std::vector<cplx>& B_buffer, const PentaDiagonalMatrix& A, const std::vector<cplx>& B, std::vector<cplx>& X) {
    int cnt = A.size;
    cplx tmp_scalar = cplx(0.0, 0.0);
    A_buffer.clear();
    std::fill(B_buffer.begin(), B_buffer.end(), cplx(0.0, 0.0));

    for (int i = 2; i <= cnt - 1; i++) {
        for (int k = i; k <= i + 1; k++) {
            tmp_scalar = (A.matrix[ptyid(k, i - 1)][ptxid(k, i - 1)] + A_buffer.matrix[ptyid(k, i - 1)][ptxid(k, i - 1)]) / (A.matrix[ptyid(i - 1, i - 1)][ptxid(i - 1, i - 1)] + A_buffer.matrix[ptyid(i - 1, i - 1)][ptxid(i - 1, i - 1)]);
            A_buffer.matrix[ptyid(k, i)][ptxid(k, i)] += -tmp_scalar * (A.matrix[ptyid(i - 1, i)][ptxid(i - 1, i)] + A_buffer.matrix[ptyid(i - 1, i)][ptxid(i - 1, i)]);
            A_buffer.matrix[ptyid(k, i + 1)][ptxid(k, i + 1)] += -tmp_scalar * (A.matrix[ptyid(i - 1, i + 1)][ptxid(i - 1, i + 1)] + A_buffer.matrix[ptyid(i - 1, i + 1)][ptxid(i - 1, i + 1)]);
            B_buffer[k - 1] += -tmp_scalar * (B[i - 1 - 1] + B_buffer[i - 1 - 1]);
        }
    }
    // i = cnt
    tmp_scalar = (A.matrix[ptyid(cnt, cnt - 1)][ptxid(cnt, cnt - 1)] + A_buffer.matrix[ptyid(cnt, cnt - 1)][ptxid(cnt, cnt - 1)]) / (A.matrix[ptyid(cnt - 1, cnt - 1)][ptxid(cnt - 1, cnt - 1)] + A_buffer.matrix[ptyid(cnt - 1, cnt - 1)][ptxid(cnt - 1, cnt - 1)]);
    A_buffer.matrix[ptyid(cnt, cnt)][ptxid(cnt, cnt)] += -tmp_scalar * (A.matrix[ptyid(cnt - 1, cnt)][ptxid(cnt - 1, cnt)] + A_buffer.matrix[ptyid(cnt - 1, cnt)][ptxid(cnt - 1, cnt)]);
    B_buffer[cnt - 1] += -tmp_scalar * (B[cnt - 1 - 1] + B_buffer[cnt - 1 - 1]);

    X[cnt - 1] = (B[cnt - 1] + B_buffer[cnt - 1]) / (A.matrix[ptyid(cnt, cnt)][ptxid(cnt, cnt)] + A_buffer.matrix[ptyid(cnt, cnt)][ptxid(cnt, cnt)]);
    X[cnt - 1 - 1] = (B[cnt - 1 - 1] + B_buffer[cnt - 1 - 1] - X[cnt - 1] * (A.matrix[ptyid(cnt - 1, cnt)][ptxid(cnt - 1, cnt)] + A_buffer.matrix[ptyid(cnt - 1, cnt)][ptxid(cnt - 1, cnt)])) /
        (A.matrix[ptyid(cnt - 1, cnt - 1)][ptxid(cnt - 1, cnt - 1)] + A_buffer.matrix[ptyid(cnt - 1, cnt - 1)][ptxid(cnt - 1, cnt - 1)]);

    for (int i = cnt - 2; i >= 1; i--) {
        X[i - 1] = ((B[i - 1] + B_buffer[i - 1]) - (X[i + 1 - 1] * (A.matrix[ptyid(i, i + 1)][ptxid(i, i + 1)] + A_buffer.matrix[ptyid(i, i + 1)][ptxid(i, i + 1)]) + X[i + 2 - 1] * (A.matrix[ptyid(i, i + 2)][ptxid(i, i + 2)] + A_buffer.matrix[ptyid(i, i + 2)][ptxid(i, i + 2)]))) /
            (A.matrix[ptyid(i, i)][ptxid(i, i)] + A_buffer.matrix[ptyid(i, i)][ptxid(i, i)]);
    }
}

#endif // __UTIL_HPP__