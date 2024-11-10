#include "../src/data_loader.h"
#include <iostream>

int main() {
    Matrix y(4, 4);
    y.init_zero();
    Matrix a(4,3);
    read_csv("./matrix_test/matrix1.csv", &a);
    Matrix b(3,4);
    read_csv("./matrix_test/matrix2.csv", &b);

    a.print_matrix();
    b.print_matrix();

    mult(&y, &a, &b);
    
    y.print_matrix();
}