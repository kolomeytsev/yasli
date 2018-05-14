#ifndef optimizers_hpp
#define optimizers_hpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "printer.hpp"


class Optimizer {
public:
    virtual void UpdateWeight(std::vector<float>* weight,
                              std::vector<float>* weight_cat,
                              const std::vector<float>& grad,
                              const std::unordered_map<uint64_t, float>& grad_cat) = 0;
};

class SGD : public Optimizer {
public:
    SGD(float lr): learning_rate(lr) {}
    void UpdateWeight(std::vector<float>* weight,
                      std::vector<float>* weight_cat,
                      const std::vector<float>& grad,
                      const std::unordered_map<uint64_t, float>& grad_cat);
private:
    float learning_rate;
};

class Adagrad : public Optimizer {
public:
    Adagrad(int size, float lr);
    void UpdateWeight(std::vector<float>* weight,
                      std::vector<float>* weight_cat,
                      const std::vector<float>& grad,
                      const std::unordered_map<uint64_t, float>& grad_cat);
private:
    std::vector<float> cumutative_gradient;
    double epsilon;
    float learning_rate;
};

#include "optimizers.cpp"
#endif // optimizers_hpp
