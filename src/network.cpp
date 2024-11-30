#include "layer.h"
#include "data_loader.h"

int main() {
    const size_t img_size = 28*28;
    const size_t train_samples_cnt = 60000;
    const size_t test_samples_cnt = 10000;
    const size_t num_classes = 10;
    const size_t batch_size = 512;

    Matrix train_data(img_size, train_samples_cnt);
   

    read_csv("./data/fashion_mnist_train_vectors.csv", &train_data);
    
    Matrix train_labels(num_classes, train_samples_cnt);

    read_csv_labels("./data/fashion_mnist_train_labels.csv", &train_labels);

    Matrix test_data(img_size, test_samples_cnt);


    read_csv("./data/fashion_mnist_test_vectors.csv", &test_data);
    
    Matrix test_labels(num_classes, test_samples_cnt);

    read_csv_labels("./data/fashion_mnist_test_labels.csv", &test_labels);

    
    //std::vector<size_t> shfl = train_data.get_shuffle_indices();
    //train_data.shuffle_cols(shfl);
    //train_labels.shuffle_cols(shfl);

    //train_data.print_matrix();
    train_data.standardize_matrix_adv();
    test_data.standardize_matrix_adv();

    //train_labels.print_matrix();

    Layer h1(&train_data, 256);
    h1.init_He();

    Layer h2(h1.output, 128);
    h2.init_He();

    Layer out(h2.output, num_classes);
    out.init_Xavier();
    float error = 0.0;

    for (size_t epoch = 0; epoch < 2; ++epoch) {
        for (size_t k = 0; k < train_data.cols; ++k) {
            h1.k = k;
            h2.k = k;
            out.k = k;
            // Forward  pass
            h1.forward(Layer::ReLU);
            h2.forward(Layer::ReLU);
            out.forward(Layer::identity);
            // std::cout << "Weights \n";
            // h1.weights->print_matrix();
            // out.weights->print_matrix();
            // std::cout << "Output \n";
            // out.output->print_matrix();

            // Error comp
            error += out.error_catCE(&train_labels)/batch_size;
            // Error diff
            out.d_catCE(&train_labels);
            out.backprop(out.output, Layer::d_identity);
            h2.backprop(h2.output, Layer::d_ReLU);
            h1.backprop(h1.output, Layer::d_ReLU, true);

            if  ((k % batch_size == 0) || (k == train_data.cols - 1)) {
                h1.update_weights();
                h2.update_weights();
                out.update_weights();
                
                std::cout << "Loss: " << error << "(epoch " << epoch << ", k " << k << ")\n";

                error = 0.0;
            }
        }
    
        if (epoch % 1 == 0) {
            //std::cout << "Loss: " << error << "(epoch " << epoch << ")\n";
        }
        
    }

    //Predict
    //h1.weights->print_matrix();
    //std::cout << "\n";
    //out.weights->print_matrix();

    h1.set_input(&test_data);
    h2.set_input(h1.output);
    out.set_input(h2.output);
    for (size_t k = 0; k < test_data.cols; ++k) {
            h1.k = k;
            h2.k = k;
            out.k = k;
            h1.forward(Layer::ReLU);
            h2.forward(Layer::ReLU);
            out.forward(Layer::identity);

            out.error_catCE(&test_labels);
    }
    //out.output->round_matrix();

    //out.output->print_matrix();
    //out.weights->print_matrix();
    std::cout << out.output->cols << "<- cols, rows ->" << out.output->rows << "\n";
    write_csv_predictions("./example_test_predictions.csv", out.output);
}
