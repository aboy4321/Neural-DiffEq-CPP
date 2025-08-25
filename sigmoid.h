#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

class Sigmoid{
    private:
        vector<double> output_vals;
    public:
        Sigmoid(){}
        vector<double> feed(const vector<double> &input);
};  
vector<double> Sigmoid::feed(const vector<double> &input){
    output_vals = vector<double>();
    for(int i = 0; i < input.size(); i++){
        output_vals.push_back(1.0 / (1.0 + exp(-input[i])));
    }
    return output_vals;
}
vector<double> Sigmoid::backprop(vector<double> grad){
    for(int o = 0; o < output_vals.size(); o++){
        grad[o] *= output_vals[o] * (1.0 - output_vals[o]);
    }
    return grad;
}
