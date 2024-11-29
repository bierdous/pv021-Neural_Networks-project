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
    Layer h1(&data, 8);
    // TODO Init weights
    h1.init_He();

    h1.weights->print_matrix();
    Layer out(h1.output, 1);
    // TODO Init weights
    out.init_Xavier();
    float error = 0.0;

    // Need atleast 1000 epochs
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        error = 0.0;
        for (size_t k = 0; k < data.cols; ++k) {
            h1.k = k;
            out.k = k;
            // Forward  pass
            h1.forward(Layer::ReLU);
            out.forward(Layer::sigmoid);
            // std::cout << "Weights \n";
            // h1.weights->print_matrix();
            // out.weights->print_matrix();
            // std::cout << "Output \n";
            // out.output->print_matrix();

            // Error comp
            error += out.error_bce(&expected);
            // Error diff
            out.d_bce(&expected);
            out.backprop(out.output, Layer::d_sigmoid);
            h1.backprop(h1.output, Layer::d_ReLU, true);
        }

        h1.update_weights();
        out.update_weights();
        
        h1.weight_diff->init_zero();
        out.weight_diff->init_zero();
        if (epoch % 10 == 0) {
            std::cout << "Loss: " << error << "(epoch " << epoch << ")\n";
        }
    }

    //Predict
    //h1.weights->print_matrix();
    //std::cout << "\n";
    //out.weights->print_matrix();
    for (size_t k = 0; k < data.cols; ++k) {
            h1.k = k;
            out.k = k;
            h1.forward(Layer::ReLU);
            out.forward(Layer::sigmoid);

            float round_out = round(out.output->get(0,k));

            std::cout << "For input: [" << data.get(0,k) << ", " << data.get(1,k) << "] predicted output is: " 
                    << round_out << " (" << out.output->get(0,k) << ")\n";
        }
}
