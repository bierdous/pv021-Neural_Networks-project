#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Data {
    public:
        size_t rows;
        size_t cols;
        float** data;

        Data(size_t row_n, size_t col_n) : rows(row_n), cols(col_n) {
            data = new float*[rows];

            for (size_t i = 0; i < rows; ++i) {
                data[i] = new float[cols];
            }
        }

        ~Data() {
            for (size_t i = 0; i < rows; ++i) {
                delete[] data[i];
            }

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
                    data[row][col] = std::stof(cell);  // Convert the cell to a float
                    col++;
                }
                row++;
            }

            file.close();

        }

        // Optional: Method to print the data for verification
        void print_data() const {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    std::cout << data[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }

        void init_rnd() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i][j] = ((float)rand())/RAND_MAX;
                }
            }
        }

        void init_zero() {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i][j] = 0;
                }
            }
        }
};

// Correct loop order
void mult(Data* y, Data* w, Data* x) {
    for (size_t i = 0; i < w->cols; ++i) {
        for (size_t j = 0; j < w->rows; ++j) {
            for (size_t k = 0; k < x->cols; ++k) {
                y->data[j][k] += w->data[j][i] * x->data[i][k];
            }
        }
    }
}