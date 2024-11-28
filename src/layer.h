#include "matrix.h"
#include <cmath>
#include <random>
#include <functional>

class Layer {
    public:
        size_t k;
        size_t input_len;
        size_t output_len;
        float learning_rate;
        Matrix* weights;
        Matrix* weight_diff;
        Matrix* input;
        Matrix* inner;
        Matrix* output;

        Layer(Matrix* in, size_t out_len, float lr = 0.1) {
            output_len = out_len;
            input_len = in->rows;

            weights = new Matrix(output_len, input_len);
            weight_diff = new Matrix(output_len, input_len);
            weight_diff->init_zero();
            input = in;
            inner = new Matrix(output_len, 1);
            output = new Matrix(output_len, in->cols);
            learning_rate = lr;
        }

        ~Layer() {
            delete weights;
            delete weight_diff;
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

        void init_Xavier() {
            // Taken from https://en.cppreference.com/w/cpp/numeric/random/normal_distribution
            std::random_device rd{};
            std::mt19937 gen{rd()};
        
            // values near the mean are the most likely
            // standard deviation affects the dispersion of generated values from the mean
            std::normal_distribution<float> d{0.0, sqrt(6.0/((float)input_len+(float)output_len))};
        
            // draw a sample from the normal distribution
            auto random_Xavier = [&d, &gen]{ return d(gen); };

            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    weights->data[j+i*weights->rows] = random_Xavier();
                }
            }
        }

        // Initializes weights with random 0-1 values
        void init_rnd() {
            std::random_device rd{};
            std::mt19937 gen{rd()};

            std::uniform_real_distribution<float> d(0.0, 1.0);
            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    weights->data[j+i*weights->rows] = d(gen);
                }
            }
        }

        static float ReLU(float inner) {
            return inner > 0.0f ? inner : 0.0f;
        }

        static float d_ReLU(float inner) {
            return inner > 0.0f ? 1.0f : 0.0f;
        }

        static float sigmoid(float inner) {
            return 1/(1+exp(-inner));
        }

        static float d_sigmoid(float inner) {
            float sig = sigmoid(inner);
            return sig*(1-sig);
        }

        void compute_inner() {
            // Zero inner
            for (size_t j = 0; j < inner->rows; ++j) {
                    inner->data[j] = 0;
            }

            // Compute inner potential
            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    inner->data[j] += weights->get(j, i) * input->get(i, k);
                }
            }
        }

        void forward(std::function<float(float)> activation_fun) {
            compute_inner();

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j + k*output->rows] = activation_fun(inner->data[j]);
            }
        }

        void forward_hidden() {
            compute_inner();

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j + k*output->rows] = ReLU(inner->data[j]);
                }
        }

        void forward_output() {
            compute_inner();

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j + k*output->rows] = sigmoid(inner->data[j]);
                }
        }

        void backprop(Matrix* dE_kdy_j, std::function<float(float)> activation_d_fun, bool is_input = false) {
            // dE_k/dw_ji or the derivative of error
            float prev_y_i_sum = 0.0;
            float dy_j_dinner = 0.0;
             for (size_t i = 0; i < weights->cols; ++i) {
                prev_y_i_sum = 0.0;
                for (size_t j = 0; j < weights->rows; ++j) {
                    dy_j_dinner = activation_d_fun(inner->data[j]);

                    prev_y_i_sum += dE_kdy_j->get(j, k) * dy_j_dinner * weights->get(j, i);

                    // dE_k/dw_ji = dE_k/dy_j * dy_j_dinner_j * dinner_j/dw_ji
                
                    weight_diff->data[j+i*weights->rows] += dE_kdy_j->get(j, k) * dy_j_dinner * input->get(i, k);
                }
                // Backpropagating dE_k/dy(k-1)_j
                // where k is the index of this layer
                // We don't want to overwrite input data.
                if (!is_input) {
                    input->data[i + k*input->rows] = prev_y_i_sum;
                }
            }
        }

        void update_weights() {
             for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    weights->data[j + i*weights->rows] -= learning_rate * weight_diff->get(j, i);
                }
            }
        }

        float error_lsq(Matrix *expected) {
            float error = 0.0; 
            for (size_t j = 0; j < output_len; ++j) {
                error += pow(output->get(j, k) - expected->get(j, k), 2);
            }
            return error/2;
        }

        void d_lsq(Matrix *expected) {
            // dE_k/dy_j = y_j - d_k
            for (int j = 0; j < output_len; j++) {
                output->data[j + k*output->rows] -= expected->get(j, k);
            }
        }

        // Assumes single output neuron
        float error_bce(Matrix *expected) {
            float d = expected->get(0, k);
            float y = output->get(0, k);
            return - (d*log(y) + (1-d)*log(1-y));
        }

        void d_bce(Matrix *exp_result) {
            float d = exp_result->get(0, k);
            float y = output->get(0, k);
            output->data[0 + k*output->rows] = (y - d) / (y*(1 - y));
        }
};