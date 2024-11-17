#include "../src/data_loader.h"
#include <iostream>

int main() {
    // --- load ----
    Matrix r(2, 3);
    r.init_rnd();

    Matrix y(4, 4);
    y.init_zero();
    
    Matrix a(4,3);
    read_csv("./matrix_test/matrix1.csv", &a);

    Matrix b(3,4);
    read_csv("./matrix_test/matrix2.csv", &b);

    Matrix c(4, 4);
    read_csv("./matrix_test/matrix3.csv", &c);

    Matrix h(28, 28); 
    read_csv("./matrix_test/matrix_heart.csv", &h);

    // ---- test ----
    std::cout << "r:\n";
    r.standardize_matrix_simple();
    r.print_matrix();
    
    std::cout << "y:\n";
    y.standardize_matrix_simple();
    y.print_matrix();

    std::cout << "a:\n";
    a.standardize_matrix_simple();
    a.print_matrix();

    std::cout << "b:\n";
    b.standardize_matrix_simple();
    b.print_matrix();

    std::cout << "c:\n";
    c.standardize_matrix_simple();
    c.print_matrix();

    std::cout << "h:\n";
    h.standardize_matrix_simple();
    h.print_matrix();
}