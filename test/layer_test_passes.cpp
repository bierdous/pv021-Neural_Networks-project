#include "../src/layer.h"
#include "../src/data_loader.h"

int main() {
    Matrix inp_m(1,1);
    inp_m.data[0] = 2;

    Layer h1(&inp_m, 1);
    h1.weights->data[0] = 2;

    Layer out(h1.output, 1);
    out.weights->data[0] = 1;
    
    h1.forward_hidden();
    out.forward_output();

    h1.output->print_matrix();
    out.output->print_matrix();

    Matrix expected(1,1);
    expected.data[0] = 2; 
    out.backprop_output(&expected);
    out.weights->print_matrix();
    h1.output->print_matrix();

    h1.init_He();
    h1.weights->print_matrix();


}
