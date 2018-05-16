#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include <unordered_map>


struct BatchIterator;

class Optimizer {
public:
    virtual ~Optimizer() {}
    virtual void UpdateWeight(std::vector<float>* weight,
                              std::vector<float>* weight_cat,
                              const std::vector<float>& grad,
                              const std::unordered_map<uint64_t, float>& grad_cat) = 0;
    virtual void UpdateWeightFtrl(std::vector<float>* weight,
                              std::vector<float>* weight_cat,
                              BatchIterator* batch_iter) = 0;
};

class SGD : public Optimizer {
public:
    SGD(float lr): learning_rate(lr) {}
    virtual ~SGD() {}
    void UpdateWeight(std::vector<float>* weight,
                      std::vector<float>* weight_cat,
                      const std::vector<float>& grad,
                      const std::unordered_map<uint64_t, float>& grad_cat);
    void UpdateWeightFtrl(std::vector<float>* weight,
                              std::vector<float>* weight_cat,
                              BatchIterator* batch_iter) {}
private:
    float learning_rate;
};

class Adagrad : public Optimizer {
public:
    Adagrad(uint64_t size, float lr);
    virtual ~Adagrad() {}
    void UpdateWeight(std::vector<float>* weight,
                      std::vector<float>* weight_cat,
                      const std::vector<float>& grad,
                      const std::unordered_map<uint64_t, float>& grad_cat);
    void UpdateWeightFtrl(std::vector<float>* weight,
                              std::vector<float>* weight_cat,
                              BatchIterator* batch_iter) {}
private:
    std::vector<float> cumutative_gradient;
    double epsilon;
    float learning_rate;
};

class Ftrl : public Optimizer {
public:
    Ftrl(float lr, float a, float b, float l1, float l2, 
            uint64_t size_num, uint64_t size_cat);
    virtual ~Ftrl() {}
    void UpdateWeight(std::vector<float>* weight,
                      std::vector<float>* weight_cat,
                      const std::vector<float>& grad,
                      const std::unordered_map<uint64_t, float>& grad_cat) {}
    void UpdateWeightFtrl(std::vector<float>* weight,
                      std::vector<float>* weight_cat,
                      BatchIterator* batch_iter);
    void UpdateWeightsNonZero(std::vector<float>* weight,
                    const std::vector<uint64_t> &non_zero_indices,
                    std::vector<float> &z_vec_p, std::vector<float> &n_vec_p);
    void UpdateParametersNonZero(std::vector<float>* weight, std::vector<float>* weight_cat,
                const std::vector<float> &data, std::vector<uint64_t> &non_zero_indices, 
                const std::vector<uint64_t> &non_zero_indices_cat, float pred, float y_true);
private:
    std::vector<float> z_vec;
    std::vector<float> n_vec;
    std::vector<float> z_vec_cat;
    std::vector<float> n_vec_cat;
    float learning_rate;
    float alpha;
    float beta;
    float lambda_first;
    float lambda_second;
};

#endif // OPTIMIZERS_H
