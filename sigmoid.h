#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>
#include <cmath>

class Sigmoid {
private:
    std::vector<double> output_vals;  // stores last forward pass outputs

public:
    Sigmoid() {}

    // Forward pass: returns activated outputs
    std::vector<double> forward(const std::vector<double>& input) {
        output_vals.resize(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            output_vals[i] = 1.0 / (1.0 + std::exp(-input[i]));
        }
        return output_vals;
    }

    // Backward pass: returns gradient w.r.t. input
    std::vector<double> backprop(const std::vector<double>& grad_output) {
        std::vector<double> grad_input(grad_output.size());
        for (size_t i = 0; i < grad_output.size(); i++) {
            grad_input[i] = grad_output[i] * output_vals[i] * (1.0 - output_vals[i]);
        }
        return grad_input;
    }
};

#endif // SIGMOID_H

