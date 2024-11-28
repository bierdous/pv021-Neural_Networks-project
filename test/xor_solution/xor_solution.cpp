#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix data(2,4);
    read_csv("./input.csv", &data);
    data.standardize_matrix_adv();
    data.print_matrix();
    Matrix expected(1,4);
    read_csv("./expected.csv", &expected);
    Matrix inp(2,1);
    
    // Using 4 neurons did not converge past a local minimum 0.5,
    // 8 neurons worked fine. 
    Layer h1(&inp, 8);
    // TODO Init weights
    h1.init_He();

    h1.weights->print_matrix();
    Layer out(h1.output, 1);
    // TODO Init weights
    out.init_rnd();
    float error = 0.0;

    // Need atleast 1000 epochs
    for (size_t epoch = 0; epoch < 10000; ++epoch) {
        error = 0.0;
        for (size_t k = 0; k < data.cols; ++k) {
            // Choose the correct sample
            data.copy_col(&inp, k);

            // Forward pass
            h1.forward(Layer::ReLU);
            out.forward(Layer::sigmoid);
            // std::cout << "Weights \n";
            // h1.weights->print_matrix();
            // out.weights->print_matrix();
            // std::cout << "Output \n";
            // out.output->print_matrix();

            // Error comp
            error += out.error_bce(&expected, k);
            // Error diff
            out.d_bce(&expected, k);
            out.backprop(out.output, Layer::d_sigmoid);
            h1.backprop(h1.output, Layer::d_ReLU);
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Loss: " << error << "(epoch " << epoch << ")\n";
        }
    }

    //Predict
    h1.weights->print_matrix();
    std::cout << "\n";
    out.weights->print_matrix();
    for (size_t k = 0; k < data.cols; ++k) {
            data.copy_col(&inp, k);
            h1.forward(Layer::ReLU);
            out.forward(Layer::sigmoid);

            float round_out = round(out.output->data[0]);

            std::cout << "For input: [" << inp.data[0] << ", " << inp.data[1] << "] predicted output is: " 
                    << round_out << " (" << out.output->data[0] << ")\n";
        }
}
