#pragma once

#include <algorithm> // For std::shuffle
#include <random>    // For std::default_random_engine
#include <cstdlib>  
#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>

class Matrix {
    public:
        size_t rows;
        size_t cols;
        float* data;

        Matrix(size_t row_n, size_t col_n) : rows(row_n), cols(col_n) {
            data = new float[rows * cols];
        }

        Matrix(Matrix *other) {
            rows = other->rows;
            cols = other->cols;

            data = new float[rows * cols];
            
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i + j * rows] = other->get(i, j);
                }
            }
        }


        ~Matrix() {
            delete[] data;
        }


        std::vector<size_t> get_shuffle_indices() {
            std::vector<size_t> column_indices(cols);
            
            for (size_t i = 0; i < cols; ++i) {
                column_indices[i] = i;
            }

            auto rd = std::random_device {}; 
            auto rng = std::default_random_engine { rd() };

            std::shuffle(std::begin(column_indices), std::end(column_indices), rng);
            
            return column_indices;
        }

        void shuffle_cols(std::vector<size_t> column_indices) {
            Matrix temp(this);

            for (size_t col = 0; col < temp.cols; col++) {
                for (size_t row = 0; row < temp.rows; row++) {
                    data[row + col*rows] = temp.get(row, column_indices[col]);
                }
            }
        }

        // Too complex, not used for now;
        std::vector<Matrix> getMinibatches(size_t minibatch_size, bool shuffle = true) {
            minibatch_size = std::min(minibatch_size, cols);

            // Generate column indices
            std::vector<size_t> column_indices(cols);
            for (size_t i = 0; i < cols; ++i) {
                column_indices[i] = i;
            }

            // Shuffle indices if required
            if (shuffle) {
                auto rng = std::default_random_engine {};
                std::shuffle(std::begin(column_indices), std::end(column_indices), rng);
            }

            // Calculate the number of minibatches
            size_t num_minibatches = std::ceil(cols / minibatch_size);

            // Create minibatches
            std::vector<Matrix> minibatches;
            minibatches.reserve(num_minibatches);

            for (size_t batch_index = 0; batch_index < num_minibatches; ++batch_index) {
                // Determine the starting and ending columns for this minibatch
                size_t start_col = batch_index * minibatch_size;
                size_t end_col = std::min(start_col + minibatch_size, cols);
                size_t current_batch_size = end_col - start_col;

                // Create a minibatch matrix
                Matrix minibatch(rows, current_batch_size);

                // Populate the minibatch with the selected columns
                for (size_t i = 0; i < current_batch_size; ++i) {
                    size_t col = column_indices[start_col + i];
                    copy_col(&minibatch, col);
                }

                minibatches.push_back(minibatch);
            }

            return minibatches;
        }

        float get(size_t row, size_t col) {
            return data[row + col * rows];
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

        void round_matrix() {
            for (size_t col = 0; col < cols; col++) {
                for (size_t row = 0; row < rows; row++) {
                    data[row + col*rows] = round(get(row, col));
                }
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
                for (size_t j = 0; j < cols; ++j)  {
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