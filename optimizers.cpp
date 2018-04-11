#include "optimizers.hpp"
#include <cmath>

#ifndef optimizers_cpp
#define optimizers_cpp

std::vector<float> SGD::UpdateWeight(const std::vector<float>& weight,
                                                                         const std::vector<float>& grad) {
    std::vector<float> new_weight(weight);
    std::vector<float> alpha_grad(grad);
    transform(alpha_grad.begin(), alpha_grad.end(),
                        alpha_grad.begin(),
                        std::bind1st(std::multiplies<float>(), learning_rate));
    
    std::transform(new_weight.begin(), new_weight.end(),
                                 alpha_grad.begin(), new_weight.begin(),
                                 std::minus<float>());
    
    return new_weight;
};


Adagrad::Adagrad(int size, float lr) {
    cumutative_gradient.resize(size);
    epsilon = 1e-8;
    learning_rate = lr;
}

std::vector<float> Adagrad::UpdateWeight(const std::vector<float>& weight,
                                                                                 const std::vector<float>& grad) {
    std::vector<float> new_weight(weight);
    std::vector<float> alpha_grad(grad);
    transform(alpha_grad.begin(), alpha_grad.end(),
                        alpha_grad.begin(),
                        std::bind1st(std::multiplies<float>(), learning_rate));

    for (int index = 0; index < weight.size(); ++index) {
        cumutative_gradient[index] += grad[index] * grad[index];
    }

    for (int index = 0; index < weight.size(); ++index) {
        alpha_grad[index] /= pow(cumutative_gradient[index] + epsilon, 0.5);
    }

    std::transform(new_weight.begin(), new_weight.end(),
                                 alpha_grad.begin(), new_weight.begin(),
                                 std::minus<float>());

    return new_weight;
};

#endif // optimizers_cpp
