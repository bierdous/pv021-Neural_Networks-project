#pragma once

#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

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

        void print_matrix(size_t r_num, size_t c_num) const {
            for (size_t i = 0; i < r_num; ++i) {
                for (size_t j = 0; j < c_num; ++j) {
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

        void init_rnd() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j*rows] = ((float)rand())/RAND_MAX;
                }
            }
        }

        void init_zero() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j *rows] = 0;
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