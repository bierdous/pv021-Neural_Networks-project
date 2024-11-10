#include <cstddef>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "matrix.h"

void read_csv(const std::string &file_path, Matrix* m, bool transpose = false) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open .csv file " << file_path << std::endl;
        return;
    }

    std::string line; 

    if (transpose) {
        size_t col = 0;
        
        while (std::getline(file, line) && col < m->cols) {
            std::stringstream lineStream(line);
            std::string cell;
            size_t row = 0;

            while (std::getline(lineStream, cell, ',') && row < m->rows) {
                m->data[row + col * m->rows] = std::stof(cell);  // Convert the cell to a float
                row++;
            }
            col++;
        }
    } else {
        size_t row = 0;

        while (std::getline(file, line) && row < m->rows) {
            std::stringstream lineStream(line);
            std::string cell;
            size_t col = 0;

            while (std::getline(lineStream, cell, ',') && col < m->cols) {
                m->data[row + col * m->rows] = std::stof(cell);  // Convert the cell to a float
                col++;
            }
            row++;
        }
    }

    file.close();
}
