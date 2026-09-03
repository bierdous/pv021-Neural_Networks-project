#!/bin/bash

echo "Adding some modules"

module add gcc-10.2

echo "#################"
echo "    COMPILING    "
echo "#################"

g++ -Wall -std=c++17 -O3 -march=native -funroll-loops src/network.cpp -o network

# With timer
# g++ -Wall -lboost_timer -std=c++17 -O3 -march=native -funroll-loops src/network.cpp -o network

echo "#################"
echo "     RUNNING     "
echo "#################"

./network
