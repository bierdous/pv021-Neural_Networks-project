#include "../src/data_loader.h"
#include <iostream>

int main() {
    Matrix r(2, 3);
    r.init_rnd();
    r.print_matrix();

    Matrix y(4, 4);
    y.init_zero();
    y.print_matrix();
    
    Matrix a(4,3);
    read_csv("./matrix_test/matrix1.csv", &a);
    a.print_matrix();

    Matrix b(3,4);
    read_csv("./matrix_test/matrix2.csv", &b);
    b.print_matrix();

    Matrix c(4, 4); 
    read_csv("./matrix_test/matrix3.csv", &c);
    c.print_matrix();

    Matrix h(28, 28); 
    read_csv("./matrix_test/matrix_heart.csv", &h);
    h.print_matrix();
}