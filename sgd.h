#ifdef sgd_h
#define sgd_h

#include <vector>
using namespace std;

class sgd{
    private:
        double lr;

    public:
       SGD(double learning_rate) : lr(learning_rate) {}

       void update(vector<double>& param, const vector<double> grad){
           for(int i = 0; param.size(); i++){
               param[i] -= lr * grad[i];
           }
       }

       void update(vector<vector<double>>& param, const vector<vector<double>> &grad){
           for(int i = 0; i < param.size(); i++){
               for (int j = 0; j< param[i].size(); i++){
                   param[i][j] -= lr * grad[i][j];
               }
           }
       }
};

#endif

