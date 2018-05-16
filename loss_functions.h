#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <vector>
#include <unordered_map>

struct BatchIterator;

class LossFunction {
public:
    virtual ~LossFunction() {}
    virtual void GetGrad(std::vector<float>* grad,
                         std::unordered_map<uint64_t, float>* grad_cat,
                         const std::vector<float>& data,
                         const std::vector<uint64_t>& indexes,
                         float label,
                         const std::vector<float>& weight,
                         const std::vector<float>& weight_cat) = 0;
    void GetBatchGrad(std::vector<float>* grad,
                      std::unordered_map<uint64_t, float>* grad_cat,
                      const BatchIterator* batch_iter,
                      const std::vector<float>& weight,
                      const std::vector<float>& weight_cat);
};

class MSE : public LossFunction {
public:
    virtual ~MSE() {}
    void GetGrad(std::vector<float>* grad,
                 std::unordered_map<uint64_t, float>* grad_cat,
                 const std::vector<float>& data,
                 const std::vector<uint64_t>& indexes,
                 float label,
                 const std::vector<float>& weight,
                 const std::vector<float>& weight_cat);
    float GetLoss(const std::vector<std::vector<float>>& data,
                  const std::vector<float>& label,
                  const std::vector<float>& weight);
};

class Logistic : public LossFunction {
public:
    virtual ~Logistic() {}
    void GetGrad(std::vector<float>* grad,
                 std::unordered_map<uint64_t, float>* grad_cat,
                 const std::vector<float>& data,
                 const std::vector<uint64_t>& indexes,
                 float label,
                 const std::vector<float>& weight,
                 const std::vector<float>& weight_cat);

    
private:
    float GetMargin(const std::vector<float>& data,
                    const std::vector<uint64_t>& indexes,
                    float label,
                    const std::vector<float>& weight,
                    const std::vector<float>& weight_cat);
    float Sigma(float x);
};

#endif // LOSS_FUNCTIONS_H
