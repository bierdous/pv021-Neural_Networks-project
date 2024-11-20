#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix inp_m(1,1);
    read_csv("./input.csv", &inp_m);

    Layer h1(&inp_m, 1);
    h1.weights->data[0] = 10;
    
    Layer out(h1.output, 1);
    out.weights->data[0] = 0.2;
    
    float error = 0.0;
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        error = 0.0;
        for (size_t k = 0; k < 1; ++k) {
            // Choose the correct sample
            out.k_in = k;

            h1.forward(Layer::ReLU);
            out.forward(Layer::sigmoid);
            std::cout << "Weights \n";
            h1.weights->print_matrix();
            out.weights->print_matrix();
            std::cout << "Output \n";
            out.output->print_matrix();

            error += out.error_lsq(&inp_m, k);
            out.d_lsq(&inp_m, k);
            out.backprop(out.output, Layer::d_sigmoid);
            h1.backprop(h1.output, Layer::d_ReLU, true);

           
        }
        
        std::cout << "Loss: " << error << "\n";
    }
}
