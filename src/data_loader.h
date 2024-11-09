#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Data {
    public:
        size_t rows;
        size_t cols;
        float* data;

        Data(size_t row_n, size_t col_n) : rows(row_n), cols(col_n) {
            data = new float[rows * cols];
        }

        ~Data() {
            delete[] data;
        }

        void read_csv(const std::string &file_path) {
            std::ifstream file(file_path);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open .csv file " << file_path << std::endl;
                return;
            }

            std::string line;
            size_t row = 0;

            while (std::getline(file, line) && row < rows) {
                std::stringstream lineStream(line);
                std::string cell;
                size_t col = 0;

                while (std::getline(lineStream, cell, ',') && col < cols) {
                    data[row + col * rows] = std::stof(cell);  // Convert the cell to a float
                    col++;
                }
                row++;
            }

            file.close();

        }

        // Optional: Method to print the data for verification
        void print_data(size_t r_num, size_t c_num) const {
            for (size_t i = 0; i < r_num; ++i) {
                for (size_t j = 0; j < c_num; ++j) {
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
};

// Correct loop order
void mult(Data* y, Data* w, Data* x) {
    for (size_t k = 0; k < x->cols; ++k) {
        for (size_t i = 0; i < w->cols; ++i) {
            for (size_t j = 0; j < w->rows; ++j) {
                y->data[j + k*y->rows] += w->data[j+i*w->rows] * x->data[i+k*x->rows];
            }
        }
    }
}