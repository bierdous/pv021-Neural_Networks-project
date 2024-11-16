#include "matrix.h"

class Layer {
    public:
        size_t input_len;
        size_t output_len;
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
        }

        ~Layer() {
            delete weights;
            delete inner;
            delete output;
        }

        float ReLU(float inner) {
            return inner > 0.0f ? inner : 0.0f;
        }

        void forward() {
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

            // Compute inner potential
            for (size_t i = 0; i < weights->cols; ++i) {
                for (size_t j = 0; j < weights->rows; ++j) {
                    inner->data[j] += weights->data[j+i*weights->rows] * input->data[i];
                }
            }

            // Apply activation function
            for (size_t j = 0; j < output->rows; ++j) {
                    output->data[j] = ReLU(inner->data[j]);
                }
        }

};