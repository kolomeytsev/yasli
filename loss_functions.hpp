#ifndef loss_functions_hpp
#define loss_functions_hpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "printer.hpp"


class LossFunction {
public:
    virtual std::vector<float> GetGrad(const std::pair<std::vector<float>, float>& object,
                                     const std::vector<float>& weight) = 0;
    virtual float GetLoss(const std::vector<std::pair<std::vector<float>, float>>& objects,
                         const std::vector<float>& weight) = 0;
};

class MSE : public LossFunction {
public:
    std::vector<float> GetGrad(const std::pair<std::vector<float>, float>& object,
                             const std::vector<float>& weight);
  
    float GetLoss(const std::vector<std::pair<std::vector<float>, float>>& objects,
                const std::vector<float>& weight);
};

class Logistic : public LossFunction {
public:
    std::vector<float> GetGrad(const std::pair<std::vector<float>, float>& object,
                             const std::vector<float>& weight);
  
    float GetLoss(const std::vector<std::pair<std::vector<float>, float>>& objects,
                const std::vector<float>& weight);

private:
    float GetMargin(const std::pair<std::vector<float>, float>& object,
                  const std::vector<float>& weight);
    float Sigma(float x);
};



#include "loss_functions.cpp"
#endif // loss_functions_hpp
