#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix train_data(4,6);
   

    read_csv("./mcc_train_data.csv", &train_data);
    Matrix train_labels(3,6);

    read_csv_labels("./mcc_train_labels.csv", &train_labels);

    train_data.standardize_matrix_adv();

    train_data.print_matrix();
    //train_labels.print_matrix();

    Layer h1(&train_data, 16);
    h1.init_He();

    Layer out(h1.output, 3);
    out.init_Xavier();
    float error = 0.0;

    // Need atleast 1000 epochs
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        error = 0.0;
        for (size_t k = 0; k < train_data.cols; ++k) {
            h1.k = k;
            out.k = k;
            // Forward  pass
            h1.forward(Layer::ReLU);
            out.forward(Layer::identity);
            // std::cout << "Weights \n";
            // h1.weights->print_matrix();
            // out.weights->print_matrix();
            // std::cout << "Output \n";
            // out.output->print_matrix();

            // Error comp
            error += out.error_catCE(&train_labels);
            // Error diff
            out.d_catCE(&train_labels);
            out.backprop(out.output, Layer::d_identity);
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
    for (size_t k = 0; k < train_data.cols; ++k) {
            h1.k = k;
            out.k = k;
            h1.forward(Layer::ReLU);
            out.forward(Layer::identity);

            out.error_catCE(&train_labels);
    }
    out.output->round_matrix();
    
    out.output->print_matrix();
    //out.weights->print_matrix();
    write_csv_predictions("mcc_train_preds.csv", out.output);
}
