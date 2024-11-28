#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix test_labels(10,10000);
    read_csv_labels("../../data/fashion_mnist_test_labels.csv", &test_labels);

    test_labels.print_matrix(0,10,0,5);

    test_labels.print_matrix(0,10,9990,10000);
}