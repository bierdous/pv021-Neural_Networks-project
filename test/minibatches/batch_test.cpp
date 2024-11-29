#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix test(6,6);

    test.init_idx();
    test.print_matrix();

    std::cout << "\n";
    std::vector<size_t> indices = test.get_shuffle_indices();
    
    for (size_t i: indices) {
        std::cout << i << ' ';
    }
    std::cout << "\n\n";

    test.shuffle_cols(indices);
    test.print_matrix();
}