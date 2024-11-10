#include "data_loader.h"
#include <iostream>

int main() {
    Matrix test_vectors(28*28, 10000);
    read_csv("../data/fashion_mnist_test_vectors.csv", &test_vectors);
    test_vectors.print_matrix(20, 1);
    test_vectors.print_matrix(1, 20);
}