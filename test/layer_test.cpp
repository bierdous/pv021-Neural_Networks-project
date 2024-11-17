#include "../src/layer.h"
#include "../src/data_loader.h"

int main() {
    Matrix inp(3,1);
    inp.data[0] = 2;
    inp.data[1] = 1;
    inp.data[2] = 4;

    Layer test(&inp,2);
    read_csv("./layer_test_data/weights1.csv", test.weights);
   

    test.forward_hidden();
    test.input->print_matrix();
    test.weights->print_matrix();
    test.inner->print_matrix();
    test.output->print_matrix();

}
