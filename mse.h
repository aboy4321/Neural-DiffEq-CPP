#ifndef mse_h
#define mse_h


#include <vector>
#include <cmath>
using namespace std;

class mse {
    public:
        static double forward(const vector<double>& y_pred, 
                           const vector<double>& y_true){
            double loss = 0.0;
            for(int i = 0; i < y_pred.size(); i++){
                double error = y_pred[i] - y_true[i];
                loss += error * error;
            }
            return loss / y_pred.size();
        }

        static vector<double> backprop(const vector<double>& y_pred, const vector<double>& y_true){
            vector<double> grad(y_pred.size(), 0.0);
            for(int i = 0; i < y_pred.size(); i++){
                grad[i] = 2.0 * (y_pred[i] - y_true[i]) / y_pred.size(); 
            }
            return grad;
        }
};

#endif
