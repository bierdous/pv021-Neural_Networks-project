#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix data(1,3);
    read_csv("./input.csv", &data);

    Matrix inp(1,1);
    

    Layer h1(&inp, 1);
    h1.weights->data[0] = 10;
    
    Layer out(h1.output, 1);
    out.weights->data[0] = 0.2;
    
    float error = 0.0;
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        error = 0.0;
        for (size_t k = 0; k < 1; ++k) {
            data.copy_col(&inp, 2);
            // Choose the correct sample
            h1.forward(Layer::ReLU);
            out.forward(Layer::sigmoid);
            std::cout << "Weights \n";
            h1.weights->print_matrix();
            out.weights->print_matrix();
            std::cout << "Output \n";
            out.output->print_matrix();

            error += out.error_lsq(&inp, k);
            out.d_lsq(&inp, k);
            out.backprop(out.output, Layer::d_sigmoid);
            h1.backprop(h1.output, Layer::d_ReLU);

           
        }
        
        std::cout << "Loss: " << error << "\n";
    }
}
