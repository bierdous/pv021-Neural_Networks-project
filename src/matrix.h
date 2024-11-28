#pragma once

#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

class Matrix {
    public:
        size_t rows;
        size_t cols;
        float* data;

        Matrix(size_t row_n, size_t col_n) : rows(row_n), cols(col_n) {
            data = new float[rows * cols];
        }

        ~Matrix() {
            delete[] data;
        }

        void copy_col(Matrix* dst, size_t k) {
            if (dst->rows != rows) {
                throw std::invalid_argument("Matrix rows don't match");
            }

            if (k > cols) {
                throw std::invalid_argument("k is larger than matrix columns");
            }

            for (size_t j = 0; j < rows; ++j) {
                dst->data[j] = data[j + k * rows];
            }
        } 

        void print_matrix(size_t r_start, size_t r_end, size_t c_start, size_t c_end) const {
            for (size_t i = r_start; i < r_end; ++i) {
                for (size_t j = c_start; j < c_end; ++j) {
                    std::cout << data[i + j * rows] << " ";
                }
                std::cout << std::endl;
            }
        }

        void print_matrix() const {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    std::cout << data[i + j * rows] << " ";
                }
                std::cout << std::endl;
            }
        }

        // preffered standardization method
        void standardize_matrix_simple() const {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j * rows] /= 255.0;
                }
            }
        }
       
        /* this standardization method should be implemented for entire dataset, not only one image. 
        Do it properly if there will be any problems with 'standardize_matrix_simple' version. */
        void standardize_matrix_adv() const {
            float mean = 0.0;
            float sum_of_squares = 0.0;
            size_t n = rows * cols; // n = total_elements

            // 1. compute mean
            for (size_t i = 0; i < n; ++i) {
                float value = data[i];
                mean += value;
                sum_of_squares += value * value;
            }
            mean /= n;

            // 2. compute st. deviation
            /*           Variance = 1/n * sum((x-mean)^2) = ... = (sum(x^2) / n) - (mean^2)
               standard deviation = sqrt(Variance) */
            float std_dev = std::sqrt((sum_of_squares / n) - (mean * mean));

            // 3. normalize the matrix
            for (size_t i = 0; i < n; ++i) {
                data[i] = (data[i] - mean) / std_dev;
            }
        }

        void init_rnd() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j * rows] = ((float)rand())/RAND_MAX;
                }
            }
        }

        void init_zero() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j * rows] = 0;
                }
            }
        }

        void init_idx() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j *rows] = i + j *rows;
                }
            }
        }
};

// Correct loop order
void mult(Matrix* y, Matrix* w, Matrix* x) {
    // Check for dimension compatibility
    if (w->cols != x->rows) {
        throw std::invalid_argument("Matrix multiplication error: Number of columns in w must equal the number of rows in x.");
    }
    if (y->rows != w->rows || y->cols != x->cols) {
        throw std::invalid_argument("Matrix multiplication error: Dimensions of y must be (w->rows, x->cols).");
    }


    for (size_t k = 0; k < x->cols; ++k) {
        for (size_t i = 0; i < w->cols; ++i) {
            for (size_t j = 0; j < w->rows; ++j) {
                y->data[j + k*y->rows] += w->data[j+i*w->rows] * x->data[i+k*x->rows];
            }
        }
    }
}