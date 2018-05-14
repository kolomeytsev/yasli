#include "optimizers.hpp"
#include <cmath>

#ifndef optimizers_cpp
#define optimizers_cpp

void SGD::UpdateWeight(std::vector<float>* weight,
                       std::vector<float>* weight_cat,
                       const std::vector<float>& grad,
                       const std::unordered_map<uint64_t, float>& grad_cat) {
    for (int index = 0; index < grad.size(); ++index) {
        (*weight)[index] -= learning_rate * grad[index];
    }

    for(auto x : grad_cat) {
        (*weight_cat)[x.first] -= learning_rate * x.second;
    }
};


Adagrad::Adagrad(int size, float lr) {
    cumutative_gradient.resize(size);
    epsilon = 1e-8;
    learning_rate = lr;
}

void Adagrad::UpdateWeight(std::vector<float>* weight,
                           std::vector<float>* weight_cat,
                           const std::vector<float>& grad,
                           const std::unordered_map<uint64_t, float>& grad_cat) {
    for (int index = 0; index < grad.size(); ++index) {
        cumutative_gradient[index] += grad[index] * grad[index];
    }

    for(auto x : grad_cat) {
        cumutative_gradient[x.first] += x.second * x.second;
    }
    
    for (int index = 0; index < grad.size(); ++index) {
        (*weight)[index] -= learning_rate * grad[index] / pow(cumutative_gradient[index] + epsilon, 0.5);
    }
    
    for(auto x : grad_cat) {
        (*weight_cat)[x.first] -= learning_rate * x.second / pow(cumutative_gradient[x.first] + epsilon, 0.5);
    }
};

#endif // optimizers_cpp
