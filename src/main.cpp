#include "data_loader.h"
#include <iostream>

int main() {
    Data test_vectors(10000, 28*28);
    test_vectors.read_csv("../data/fashion_mnist_test_vectors.csv");
    std::cout << test_vectors.rows;
    test_vectors.print_data();
}