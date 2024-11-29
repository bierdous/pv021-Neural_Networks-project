#include "../../src/layer.h"
#include "../../src/data_loader.h"

int main() {
    Matrix test_labels(10,10000);
    //read_csv_labels("../../data/fashion_mnist_test_labels.csv", &test_labels);

    //test_labels.print_matrix(0,10,0,5);

    //test_labels.print_matrix(0,10,9990,10000);
    Matrix test_preds(10,5);
    read_csv("./test_preds.csv", &test_preds);
    test_preds.print_matrix();
    write_csv_predictions("./test_labels_out.csv", &test_preds);

}