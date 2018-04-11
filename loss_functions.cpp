#include "loss_functions.hpp"

#ifndef loss_functions_cpp
#define loss_functions_cpp

std::vector<float> MSE::GetGrad(const std::pair<std::vector<float>, float>& object,
                                                            const std::vector<float>& weight) {
    std::vector<float> data(object.first);
    float label = object.second;

    std::vector<float> grad;
    grad.resize(data.size());

    float coef = 2 * (std::inner_product(weight.begin(), weight.end(), data.begin(), 0.0) - label);

    std::transform(data.begin(), data.end(),
                                 grad.begin(),
                                 std::bind1st(std::multiplies<float>(), coef));

    return grad;
}

float MSE::GetLoss(const std::vector<std::pair<std::vector<float>, float>>& objects,
                                 const std::vector<float>& weight) {
    float error = 0;
    for (const auto& pair : objects) {
        std::vector<float> data = pair.first;
        float label = pair.second;
        error += pow(std::inner_product(data.begin(), data.end(), weight.begin(), 0.0) - label, 2);
    }
    return error;
}


float Logistic::GetMargin(const std::pair<std::vector<float>, float>& object,
                                                    const std::vector<float>& weight) {
    return std::inner_product(weight.begin(), weight.end(), object.first.begin(), 0.0) * object.second;
}

float Logistic::Sigma(float x) {
    return 1.0 / (1 + exp(-1 * x));
}

std::vector<float> Logistic::GetGrad(const std::pair<std::vector<float>, float>& object,
                                                                         const std::vector<float>& weight) {
    std::vector<float> data(object.first);
    float label = object.second;

    std::vector<float> grad;
    grad.resize(data.size());

    float coef = -1 * label * Sigma(-1 * GetMargin(object, weight));
    
    std::transform(data.begin(), data.end(),
                    grad.begin(),
                    std::bind1st(std::multiplies<float>(), coef));

    return grad;
}

float Logistic::GetLoss(const std::vector<std::pair<std::vector<float>, float>>& objects,
                                                const std::vector<float>& weight) {
    return 0;
}


#endif // loss_functions_cpp
