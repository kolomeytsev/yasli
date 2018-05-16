#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

#include "loss_functions.h"
#include "data_reader.h"

void LossFunction::GetBatchGrad(std::vector<float>* grad,
                                std::unordered_map<uint64_t, float>* grad_cat,
                                const BatchIterator* batch_iter,
                                const std::vector<float>& weight,
                                const std::vector<float>& weight_cat) {
    size_t batch_size = batch_iter->size;

    auto data_it = batch_iter->data_it;
    auto data_cat_it = batch_iter->data_cat_it;
    auto target_it = batch_iter->target_it;
    for (size_t index = 0; index < batch_size; ++index) {
        GetGrad(grad, grad_cat, *data_it, *data_cat_it, *target_it, weight, weight_cat);
        ++data_it;
        ++data_cat_it;
        ++target_it;
    }
    std::transform(grad->begin(), grad->end(), grad->begin(),
                   std::bind1st(std::multiplies<float>(), 1.0 / batch_size));
    for(auto x : *grad_cat) {
        (*grad_cat)[x.first] *= 1.0 / batch_size;
    }
}

void MSE::GetGrad(std::vector<float>* grad,
                  std::unordered_map<uint64_t, float>* grad_cat,
                  const std::vector<float>& data,
                  const std::vector<uint64_t>& indexes,
                  float label,
                  const std::vector<float>& weight,
                  const std::vector<float>& weight_cat) {
    float coef = std::inner_product(weight.begin(), weight.end(), data.begin(), 0.0);

    for(auto x : *grad_cat) {
        coef += weight_cat[x.first];
    }

    coef = 2 * (coef - label);

    std::transform(data.begin(), data.end(),
                   grad->begin(),
                   std::bind1st(std::multiplies<float>(), coef));
    for (int index : indexes) {
        int count = grad_cat->count(index);
        if (count) {
            (*grad_cat)[index] += coef;
        } else {
            (*grad_cat)[index] = coef;
        }
    }
}

//float MSE::GetLoss(const std::vector<std::vector<float>>& data,
//                   const std::vector<float>& label,
//                   const std::vector<float>& weight) {
//    float error = 0;
//    for (int index = 0; index < label.size(); ++index) {
//        error += pow(std::inner_product(
//            data[index].begin(), data[index].end(), weight.begin(), 0.0) - label[index], 2);
//    }
//    return error;
//}

float Logistic::GetMargin(const std::vector<float>& data,
                          const std::vector<uint64_t>& indexes,
                          float label,
                          const std::vector<float>& weight,
                          const std::vector<float>& weight_cat) {
    float coef = std::inner_product(weight.begin(), weight.end(), data.begin(), 0.0);
    for(int index : indexes) {
        coef += weight_cat[index];
    }
    return coef * label;
}

float Logistic::Sigma(float x) {
    return 1.0 / (1 + exp(-1 * x));
}

void Logistic::GetGrad(std::vector<float>* grad,
                       std::unordered_map<uint64_t, float>* grad_cat,
                       const std::vector<float>& data,
                       const std::vector<uint64_t>& indexes,
                       float label,
                       const std::vector<float>& weight,
                       const std::vector<float>& weight_cat) {
    float coef = -1 * label * Sigma(-1 * GetMargin(data, indexes, label, weight, weight_cat));
    std::transform(data.begin(), data.end(),
                   grad->begin(),
                   std::bind1st(std::multiplies<float>(), coef));
    for (int index : indexes) {
        int count = grad_cat->count(index);
        if (count) {
            (*grad_cat)[index] += coef;
        } else {
            (*grad_cat)[index] = coef;
        }
    }
}
