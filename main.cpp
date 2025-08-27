#include <iostream>
#include <vector>
#include "mse.h"   // include your MSE class directly

int main() {
    mse mse;

    std::vector<double> y_true = {1.0, 2.0, 3.0};
    std::vector<double> y_pred = {1.5, 2.5, 2.0};

    double loss = mse.feed(y_true, y_pred);
    std::cout << "MSE Loss = " << loss << std::endl;

    std::vector<double> grad = mse.backprop(y_true, y_pred);
    std::cout << "Gradients: ";
    for (double g : grad) {
        std::cout << g << " ";
    }
    std::cout << std::endl;

    return 0;
}

