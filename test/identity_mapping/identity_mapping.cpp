#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix inp_m(1,1);
    read_csv("./input.csv", &inp_m);

    Layer h1(&inp_m, 1);
    h1.weights->data[0] = 30;
    
    Layer out(h1.output, 1);
    out.weights->data[0] = 30;
    
    float error = 0.0;
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        error = 0.0;
        for (size_t k = 0; k < 1; ++k) {
            // Choose the correct sample
            out.k_in = k;

            h1.forward_hidden();
            out.forward_output();
            std::cout << "Weights \n";
            h1.weights->print_matrix();
            out.weights->print_matrix();

            error += out.error_lsq(&inp_m, k);
            std::cout << "Output \n";
            out.output->print_matrix();
            out.backprop_output(&inp_m, k);
            h1.backprop_hidden(&out, true);

           
        }
        
        std::cout << "Loss: " << error << "\n";
    }
}
