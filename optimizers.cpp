#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>

#include "optimizers.h"
#include "data_reader.h"

void SGD::UpdateWeight(std::vector<float>* weight,
                       std::vector<float>* weight_cat,
                       const std::vector<float>& grad,
                       const std::unordered_map<uint64_t, float>& grad_cat) {
    for (uint64_t index = 0; index < grad.size(); ++index) {
        (*weight)[index] -= learning_rate * grad[index];
    }

    for(auto x : grad_cat) {
        (*weight_cat)[x.first] -= learning_rate * x.second;
    }
}

Adagrad::Adagrad(uint64_t size, float lr) {
    cumutative_gradient.resize(size);
    epsilon = 1e-8;
    learning_rate = lr;
}

void Adagrad::UpdateWeight(std::vector<float>* weight,
                           std::vector<float>* weight_cat,
                           const std::vector<float>& grad,
                           const std::unordered_map<uint64_t, float>& grad_cat) {
    for (uint64_t index = 0; index < grad.size(); ++index) {
        cumutative_gradient[index] += grad[index] * grad[index];
    }

    for(auto x : grad_cat) {
        cumutative_gradient[x.first] += x.second * x.second;
    }
    
    for (uint64_t index = 0; index < grad.size(); ++index) {
        (*weight)[index] -= learning_rate * grad[index] / pow(cumutative_gradient[index] + epsilon, 0.5);
    }
    
    for(auto x : grad_cat) {
        (*weight_cat)[x.first] -= learning_rate * x.second / pow(cumutative_gradient[x.first] + epsilon, 0.5);
    }
}

Ftrl::Ftrl(float lr, float a, float b, float l1, float l2, uint64_t size_num, uint64_t size_cat) {
    learning_rate = lr;
    alpha = a;
    beta = b;
    lambda_first = l1;
    lambda_second = l2;
    z_vec.resize(size_num, 0);
    n_vec.resize(size_num, 0);
    z_vec_cat.resize(size_cat, 0);
    n_vec_cat.resize(size_cat, 0);
}

float Sign(float x) {
    if (x < 0) {
        return -1.0f;
    } else {
        return 1.0f;
    }
}

void Ftrl::UpdateWeightsNonZero(std::vector<float>* weight,
                    const std::vector<uint64_t> &non_zero_indices,
                    std::vector<float> &z_vec_p, std::vector<float> &n_vec_p) {
    for (auto it = non_zero_indices.begin(); it != non_zero_indices.end(); ++it) {
        uint64_t cur_index = *it;
        if (abs(z_vec_p[cur_index]) <= lambda_first) {
            (*weight)[cur_index] = 0.0f;
        } else {
            float multiplier = -1.0f / ((beta + sqrt(n_vec_p[cur_index])) / alpha + lambda_second);
            (*weight)[cur_index] = multiplier * (z_vec_p[cur_index] - 
                                        Sign(z_vec_p[cur_index]) * lambda_first);
        }
    }
}

void Ftrl::UpdateParametersNonZero(std::vector<float>* weight, std::vector<float>* weight_cat,
                const std::vector<float> &data, std::vector<uint64_t> &non_zero_indices, 
                const std::vector<uint64_t> &non_zero_indices_cat, float pred, float y_true) {
    float g, sigma;
    float y_true_transformed = (y_true + 1) / 2; // {-1, 1} -> {0, 1}
    float grad = pred - y_true_transformed;
    for (auto it = non_zero_indices.begin(); it != non_zero_indices.end(); ++it) {
        g = grad * data[*it];
        sigma = 1.0 / alpha * (sqrt(n_vec[*it] + g * g) - sqrt(n_vec[*it]));
        z_vec[*it] = z_vec[*it] + g - sigma * (*weight)[*it];
        n_vec[*it] += g * g;
    }
    for (auto it = non_zero_indices_cat.begin(); it != non_zero_indices_cat.end(); ++it) {
        g = grad * 1;
        sigma = 1.0 / alpha * (sqrt(n_vec_cat[*it] + g * g) - sqrt(n_vec_cat[*it]));
        z_vec_cat[*it] = z_vec_cat[*it] + g - sigma * (*weight_cat)[*it];
        n_vec_cat[*it] += g * g;
    }
}

void Ftrl::UpdateWeightFtrl(std::vector<float>* weight,
                       std::vector<float>* weight_cat,
                       BatchIterator* batch_iter) {
    uint64_t batch_size = batch_iter->size;

    auto data_it = batch_iter->data_it;
    auto data_cat_it = batch_iter->data_cat_it;
    auto target_it = batch_iter->target_it;
    for (uint64_t batch_index = 0; batch_index < batch_size; ++batch_index) {

        std::vector<uint64_t> non_zero_indices;
        uint64_t index = 0;
        for (auto it=data_it->begin(); it != data_it->end(); ++it) {
            if (*it != 0) {
                non_zero_indices.push_back(index);
            }
            index++;
        }

        UpdateWeightsNonZero(weight, non_zero_indices, z_vec, n_vec);
        UpdateWeightsNonZero(weight_cat, *data_cat_it, z_vec_cat, n_vec_cat);

        float coef = std::inner_product(weight->begin(), weight->end(), data_it->begin(), 0.0);
        for (auto it=data_cat_it->begin(); it != data_cat_it->end(); ++it) {
            coef += (*weight_cat)[*it];
        }
        float pred = 1.0f / (1.0f + exp( - coef));
        UpdateParametersNonZero(weight, weight_cat, *data_it, 
                                non_zero_indices, *data_cat_it, pred, *target_it);
        ++data_it;
        ++data_cat_it;
        ++target_it;
    }
}
