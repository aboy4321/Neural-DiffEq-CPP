#include <vector>
#include <cstdlib>
using namespace std;

class Linear{
    private:
        int input_dim, output_dim;
        vector<double> input_vals, output_vals;
        double eta;
        vector<vector<double>> weights;
    public:
        Linear(){}
        Linear(int input_size, int output_size, double learning_rate);

        vector<double> forward(const vector<double> &input);
        vector<double> backprop(const vector<double> &grad);
};
Linear::Linear(int input_size, int output_size, double learning_rate){
    output_dim = output_size;
    input_dim = input_size;
    eta = learning_rate;

    weights.resize(output_dim); 
    for(int o = 0; o < output_size; o++){
        weights[o].resize(input_dim + 1);
        for(int i = 0; i < input_size + 1; i++){
            weights[o][i] = (double)rand() / RAND_MAX - 0.5;
        }
    }
}
vector<double> Linear::forward(const vector<double> &input){
    output_vals.resize(output_dim); 
    input_vals = input;

    for(int o = 0; o < output_dim; o++){
        double sum = 0.0;
        for(int w = 0; w < input_dim; w++){
            sum += weights[o][w] * input[w];
        }
        sum += weights[o][input_dim];
        output_vals[o] = sum;
    }
    return output_vals;
}
vector<double> Linear::backprop(const vector<double> &grad){
    vector<double> prev_grad;

    for(int i = 0; i < input_dim; i++){
        double g = 0.0;
        for(int o = 0; o < output_dim; o++){
            g += (grad[o] * weights[o][i]);
        }
        prev_grad.push_back(g);
    }
    for(int o = 0; o < output_dim; o++){
        for(int i = 0; i < input_dim; i++){
            weights[o][i] -= (eta * grad[o] * input_vals[i]);
        }
        weights[o][input_dim] -= eta * grad[o];
    }
    return prev_grad;
}
