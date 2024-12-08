/**
 * Implementation of an MLP layer.
 * 
 * Authors: Marek Dohnal (552405), Martin Bertko (514588)
 * Date: 07/12/2024
 */

#include "matrix.h"
#include <cmath>
#include <random>
#include <functional>

class Layer {
    public:
        size_t k; // Index of a current sample
        size_t input_len;
        size_t output_len;
        float learning_rate;
        Matrix* weights;
        Matrix* weight_diff; // Weights updated by backpropagation
        Matrix* input;
        Matrix* inner;
        Matrix* output;

        /**
         * Initializes a layer by passing a pointer to an input matrix,
         * which is an output of the previous layer, a number of neurons in this layer,
         * and a learning rate.
         * @param in Pointer to a matrix, which is an output of the previous layer.
         * @param out_len Number of neurons in this layer.
         * @param lr The learning rate used in this layer.
         */
        Layer(Matrix* in, size_t out_len, float lr = 0.0002) {
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
        
        /**
         * Sets a different input matrix to the layer.
         * Used when switching from training to testing.
         * @param m A matrix to be set as new input.
         */
        void set_input(Matrix *m) {
            weight_diff->init_zero();
            input = m;
            output->cols = m->cols;
        }

        /**
         * He initialization of weights used for the ReLU layers.
         */
        void init_He() {
            // Taken from https://en.cppreference.com/w/cpp/numeric/random/normal_distribution
            //std::random_device rd{};
            unsigned int seed = 0;
            std::mt19937 gen{seed};
        
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

        /**
         * Xavier initialization of weights used for the identity/softmax layers.
         */
        void init_Xavier() {
            // Taken from https://en.cppreference.com/w/cpp/numeric/random/normal_distribution
            //std::random_device rd{};
            unsigned int seed = 0;
            std::mt19937 gen{seed};
        
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

        /**
         * Random weights initialization.
         */
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

        /**
         * ReLU activation function
         * @param inner The inner potential of a neuron.
         * @return The ReLU activation.
         */
        static float ReLU(float inner) {
            return inner > 0.0f ? inner : 0.0f;
        }

        /**
         * Derivative of the ReLU activation function
         * @param inner The inner potential of a neuron.
         * @return The derivative of ReLU activation.
         */
        static float d_ReLU(float inner) {
            return inner > 0.0f ? 1.0f : 0.0f;
        }

        /**
         * Sigmoid activation function.
         * @param inner The inner potential of a neuron.
         * @return The sigmoid activation.
         */
        static float sigmoid(float inner) {
            return 1/(1+exp(-inner));
        }

        /**
         * Derivative of the sigmoid activation function.
         * @param inner The inner potential of a neuron.
         * @return The derivative of sigmoid activation.
         */
        static float d_sigmoid(float inner) {
            float sig = sigmoid(inner);
            return sig*(1-sig);
        }

        /**
         * Identity activation function. Used for an output layer,
         * for which the error is calculated via categorical cross entropy,
         * where softmax is already included.
         * @param inner The inner potential of a neuron.
         * @return The inner potential of a neuron.
         */
        static float identity(float inner) {
            return inner;
        }

        /**
         * Identity activation function. Used for backpropagation in an output layer,
         * for which the error is calculated via categorical cross entropy,
         * where softmax is already included. 
         * @param inner The inner potential of a neuron.
         * @return 1
         */
        static float d_identity(float inner) {
            return 1;
        }

        /**
         * Computes the inner potential of a layer.
         */
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

        /**
         * Implements a forward pass using an activation function.
         * @param activation_fun An activation function to be used.
         */
        void forward(std::function<float(float)> activation_fun) {
            compute_inner();

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j + k*output->rows] = activation_fun(inner->data[j]);
            }
        }

        /**
         * Implements backpropagation.
         * @param dE_kdy_j A matrix containing losses based on input of the previous layer.
         * @param activation_d_fun A derivative of an activation function used in the forward pass.
         * @param is_input True, when the input to this layer is the input layer 
         * (Losses should not be passed, because the input would be overwritten). False otherwise.
         */
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

        /**
         * Updates weights. The update values are contained in weight_diff, and are given by backpropagation.
         */
        void update_weights() {
            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    weights->data[j + i*weights->rows] -= learning_rate * weight_diff->get(j, i);
                }
            }
            weight_diff->init_zero();
        }

        /**
         * Least squares error function.
         * @param expected The matrix of expected values.
         */
        float error_lsq(Matrix *expected) {
            float error = 0.0; 
            for (size_t j = 0; j < output_len; ++j) {
                error += pow(output->get(j, k) - expected->get(j, k), 2);
            }
            return error/2;
        }

        /**
         * Derivative of least squares error.
         * @param expected The matrix of expected values.
         */
        void d_lsq(Matrix *expected) {
            // dE_k/dy_j = y_j - d_k
            for(size_t j = 0; j < output_len; j++) {
                output->data[j + k*output->rows] -= expected->get(j, k);
            }
        }

        /**
         * Categorical cross entropy, containing softmax.
         * @param expected The matrix of expected values.
         */
        float error_catCE(Matrix *expected) {
            float error = 0.0;

            float denom_sum = 0.0;
            for (size_t j = 0; j < output->rows; ++j) {
                denom_sum += exp(output->get(j, k));
            }

            for (size_t j = 0; j < expected->rows; ++j) {
                output->data[j + k*output->rows] = exp(output->get(j, k))/denom_sum;

                float target_class = expected->get(j, k);

                if (target_class == 1) {
                    error = -log(output->get(j,k));
                }
            }
            return error;
        }

        /**
         * Derivative of categorical cross entropy and softmax.
         * Backprop should be called with d_identity() activation function (which is neutral)
         * As this derivative already computes the derivative of error based on inner potential.
         * @param expected The matrix of expected values.
         */
        void d_catCE(Matrix *expected) {
            for (size_t j = 0; j < expected->rows; ++j) {
                float target_class = expected->get(j, k);

                if (target_class == 1) {
                    output->data[j + k * output->rows] = output->get(j,k) - 1;
                }
            }
        }

        /**
         * Binary cross entropy.
         * @param expected The matrix of expected values.
         */
        float error_bce(Matrix *expected) {
            float d = expected->get(0, k);
            float y = output->get(0, k);
            return - (d*log(y) + (1-d)*log(1-y));
        }

        /**
         * Derivative of binary cross entropy.
         * @param expected The matrix of expected values.
         */
        void d_bce(Matrix *expected) {
            float d = expected->get(0, k);
            float y = output->get(0, k);
            output->data[0 + k*output->rows] = (y - d) / (y*(1 - y));
        }
};