#include <iostream>
#include <vector>
#include "linear.h"
#include "sigmoid.h"
#include "mse.h"

int main() {
    // --- 1. Network components ---
    Linear linear(2, 1, 0.1);   // 2 inputs → 1 output
    Sigmoid sigmoid;

    // --- 2. Simple dataset (XOR) ---
    std::vector<std::vector<double>> X = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<double> Y = {0, 1, 1, 0};

    // --- 3. Training loop ---
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;

        for (int i = 0; i < X.size(); i++) {
            // --- Forward pass ---
            std::vector<double> z = linear.forward(X[i]);
            std::vector<double> y_pred = sigmoid.forward(z);

            // --- Compute loss ---
            double loss = mse::forward(y_pred, {Y[i]});
            epoch_loss += loss;

            // --- Backward pass ---
            std::vector<double> grad_loss = mse::backprop(y_pred, {Y[i]});
            std::vector<double> grad_sigmoid = sigmoid.backprop(grad_loss);
            linear.backprop(grad_sigmoid);
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss << std::endl;
        }
    }

    // --- 4. Test predictions ---
    std::cout << "\nPredictions after training:\n";
    for (int i = 0; i < X.size(); i++) {
        std::vector<double> out = sigmoid.forward(linear.forward(X[i]));
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] -> ";
        std::cout << "Pred: " << out[0] << ", Target: " << Y[i] << std::endl;
    }

    return 0;
}

