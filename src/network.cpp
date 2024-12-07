/**
 * Implementation of an MLP.
 * 
 * Authors: Marek Dohnal (552405), Martin Bertko (514588)
 * Date: 07/12/2024
 */

#include "layer.h"
#include "data_loader.h"

int main() {
    // Metadata
    const size_t img_size = 28*28;
    const size_t train_samples_cnt = 60000;
    const size_t test_samples_cnt = 10000;
    const size_t num_classes = 10;

    // Hyperparameters
    const size_t batch_size = 512;
    const float learning_rate = 0.0002;
    const size_t epoch_cnt = 9;

    // Loading training data and labels
    Matrix train_data(img_size, train_samples_cnt);
    read_csv("./data/fashion_mnist_train_vectors.csv", &train_data);
    
    Matrix train_labels(num_classes, train_samples_cnt);
    read_csv_labels("./data/fashion_mnist_train_labels.csv", &train_labels);

    // Loading testing data and labels
    Matrix test_data(img_size, test_samples_cnt);
    read_csv("./data/fashion_mnist_test_vectors.csv", &test_data);
    
    Matrix test_labels(num_classes, test_samples_cnt);
    read_csv_labels("./data/fashion_mnist_test_labels.csv", &test_labels);

    // Standardization
    train_data.standardize_matrix_adv();
    test_data.standardize_matrix_adv();

    // Layers and weights initialization
    Layer h1(&train_data, 256, learning_rate);
    h1.init_He();

    Layer h2(h1.output, 128, learning_rate);
    h2.init_He();

    Layer out(h2.output, num_classes, learning_rate);
    out.init_Xavier();

    // Training
    float error = 0.0;
    float total_epoch_loss = 0.0;
    for (size_t epoch = 0; epoch < epoch_cnt; ++epoch) {
        // Data shuffling
            //std::vector<size_t> shfl = train_data.get_shuffle_indices();
            //train_data.shuffle_cols(shfl);
            //train_labels.shuffle_cols(shfl);

        for (size_t k = 0; k < train_data.cols; ++k) {
            h1.k = k;
            h2.k = k;
            out.k = k;
            // Forward  pass
            h1.forward(Layer::ReLU);
            h2.forward(Layer::ReLU);
            out.forward(Layer::identity);

            // Error comp
            error += out.error_catCE(&train_labels)/batch_size;
            
            // Backpropadation
            out.d_catCE(&train_labels);
            out.backprop(out.output, Layer::d_identity);
            h2.backprop(h2.output, Layer::d_ReLU);
            h1.backprop(h1.output, Layer::d_ReLU, true);

            // Weights update
            if  ((k % batch_size == 0) || (k == train_data.cols - 1)) {
                h1.update_weights();
                h2.update_weights();
                out.update_weights();
                
                total_epoch_loss += error;

                error = 0.0;
            }
        }
        float avg_epoch_loss = total_epoch_loss/(train_samples_cnt/(float)batch_size);
        std::cout << "Loss: " << avg_epoch_loss << "(epoch " << epoch << ")\n";
        total_epoch_loss = 0.0;
    }

    // Prediction on train data

    for (size_t k = 0; k < train_data.cols; ++k) {
            h1.k = k;
            h2.k = k;
            out.k = k;
            h1.forward(Layer::ReLU);
            h2.forward(Layer::ReLU);
            out.forward(Layer::identity);
    }
    write_csv_predictions("./train_predictions.csv", out.output);

    // Prediction on test data

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
    }
    write_csv_predictions("./test_predictions.csv", out.output);
}
