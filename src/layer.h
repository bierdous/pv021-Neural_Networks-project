#include "matrix.h"
#include <cmath>
#include <random>

class Layer {
    public:
        size_t k_in;
        size_t input_len;
        size_t output_len;
        float learning_rate;
        Matrix* weights;
        Matrix* input;
        Matrix* inner;
        Matrix* output;

        Layer(Matrix* in, size_t out_len) {
            output_len = out_len;
            input_len = in->rows;
            weights = new Matrix(output_len, input_len);
            input = in;
            inner = new Matrix(output_len, 1);
            output = new Matrix(output_len, 1);
            learning_rate = 0.01;
            k_in = 0;
            // Check for dimension compatibility
            if (weights->cols != input->rows) {
                throw std::invalid_argument("Matrix multiplication error: Number of columns in w must equal the number of rows in x.");
            }
            if (output->rows != weights->rows || output->cols != input->cols) {
                throw std::invalid_argument("Matrix multiplication error: Dimensions of y must be (w->rows, x->cols).");
            }

            if (inner->rows != weights->rows || inner->cols != input->cols) {
                throw std::invalid_argument("Matrix multiplication error: Dimensions of y must be (w->rows, x->cols).");
            }
        }

        ~Layer() {
            delete weights;
            delete inner;
            delete output;
        }
        
        void init_He() {
            // Taken from https://en.cppreference.com/w/cpp/numeric/random/normal_distribution
            std::random_device rd{};
            std::mt19937 gen{rd()};
        
            // values near the mean are the most likely
            // standard deviation affects the dispersion of generated values from the mean
            std::normal_distribution<float> d{0.0, sqrt(2.0/(float)input_len)};
        
            // draw a sample from the normal distribution
            auto random_He = [&d, &gen]{ return d(gen); };

            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    weights->data[j+i*weights->rows] = random_He();
                }
            }
        }

        float ReLU(float inner) {
            return inner > 0.0f ? inner : 0.0f;
        }

        float d_ReLU(float inner) {
            return inner > 0.0f ? 1.0f : 0.0f;
        }

        float sigmoid(float inner) {
            return 1/(1+exp(-inner));
        }

        float d_sigmoid(float inner) {
            float sig = sigmoid(inner);
            return sig*(1-sig);
        }

        void compute_inner() {
            // Compute inner potential
            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    inner->data[j] += weights->data[j+i*weights->rows] * input->data[i + k_in*input_len];
                }
            }
        }

        void forward_hidden() {
            compute_inner();

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j] = ReLU(inner->data[j]);
                }
        }

        void forward_output() {
            compute_inner();

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j] = sigmoid(inner->data[j]);
                }
        }
    
        void backprop_output(Matrix* exp_result) {
            
            // dE_k/dy_j = y_j - d_k
            for (int j = 0; j < output_len; j++) {
                output->data[j] -= exp_result->data[j];
            }

            // dE_k/dw_ji
            float prev_y_i_sum = 0.0;
            float dy_j_dinner = 0.0;
             for (size_t i = 0; i < weights->cols; ++i) {
                prev_y_i_sum = 0.0;
                for (size_t j = 0; j < weights->rows; ++j) {
                    dy_j_dinner = d_sigmoid(inner->data[j]);

                    prev_y_i_sum += output->data[j] * dy_j_dinner * weights->data[j+i*weights->rows];

                    // dE_k/dw_ji = dE_k/dy_j * dy_j_dinner_j * dinner_j/dw_ji
                
                    weights->data[j+i*weights->rows] -= learning_rate * output->data[j] * dy_j_dinner * input->data[i + k_in*input_len];
                }
                // Backpropagating dE_k/dy(k-1)_j
                // where k is the index of this layer
                input->data[i + k_in*input_len] = prev_y_i_sum;
            }
        }

        void backprop_hidden(Matrix* dEk_dyj_last, Layer* previous, bool is_input) {
            
             // dE_k/dw_ji
            float prev_y_i_sum = 0.0;
            float dy_j_dinner = 0.0;
             for (size_t i = 0; i < weights->cols; ++i) {
                prev_y_i_sum = 0.0;
                for (size_t j = 0; j < weights->rows; ++j) {
                    dy_j_dinner = d_ReLU(inner->data[j]);

                    prev_y_i_sum += output->data[j] * dy_j_dinner * weights->data[j+i*weights->rows];

                    // dE_k/dw_ji = dE_k/dy_j * dy_j_dinner_j * dinner_j/dw_ji
                    // dE_k/dy_j is taken from previous layer because output->data[j] == input->data[i] (previous layer)
                
                    weights->data[j+i*weights->rows] -= learning_rate * output->data[j] * dy_j_dinner * input->data[k_in*input_len];
                }

                // We don't want to overwrite input data.
                if (!is_input) {
                    previous->output->data[i] = prev_y_i_sum;
                }
            }
        }

        float error_lsq(Matrix *expected) {
            float error = 0.0; 
            for (size_t j = 0; j < output_len; ++j) {
                error += pow(output->data[j] - expected->data[j], 2);
            }
            return error/2;
        }
};