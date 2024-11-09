#include <boost/timer/timer.hpp>
#include "data_loader.h"
#include <iostream>

#include <cmath>

int main() {
    Data y(784,128);
    y.init_zero();
    Data w(128,128);
    w.init_rnd();
    Data x(784,128);
    x.init_rnd();

    boost::timer::auto_cpu_timer t;

    for (int i = 0; i < 100; i++) {
        mult(&y, &w, &x);
    }
}