#ifndef optimizers_hpp
#define optimizers_hpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include "printer.hpp"

class Optimizer {
public:
    virtual std::vector<float> UpdateWeight(const std::vector<float>& weight,
                                           const std::vector<float>& grad) = 0;
};

class SGD : public Optimizer {
public:
    SGD(float lr): learning_rate(lr) {}
    std::vector<float> UpdateWeight(const std::vector<float>& weight,
                                  const std::vector<float>& grad);
private:
    float learning_rate;
};

class Adagrad : public Optimizer {
public:
    Adagrad(int size, float lr);
    std::vector<float> UpdateWeight(const std::vector<float>& weight,
                                  const std::vector<float>& grad);
private:
    std::vector<float> cumutative_gradient;
    double epsilon;
    float learning_rate;
};

#include "optimizers.cpp"
#endif // optimizers_hpp
