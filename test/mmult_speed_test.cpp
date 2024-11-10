#include <boost/timer/timer.hpp>
#include "../src/matrix.h"
#include <iostream>

// Has to be ran like this:
// g++ -Wall -lboost_timer -std=c++17 -O3 -march=native -funroll-loops test/mmult_speed_test.cpp -o out/mmult_speed_test
// out/mmult_speed_test
int main() {
    Matrix y(128, 784);
    y.init_zero();
    Matrix w(128,128);
    w.init_rnd();
    Matrix x(128, 784);
    x.init_rnd();

    boost::timer::auto_cpu_timer t;

    for (int i = 0; i < 100; i++) {
        mult(&y, &w, &x);
    }
}