#ifndef loss_functions_hpp
#define loss_functions_hpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "printer.hpp"
#include "data_reader.hpp"
#include <unordered_map>


class LossFunction {
public:
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
//    virtual float GetLoss(const std::vector<std::vector<float>>& data,
//                          const std::vector<float>& label,
//                          const std::vector<float>& weight) = 0;
};

class MSE : public LossFunction {
public:
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
    void GetGrad(std::vector<float>* grad,
                 std::unordered_map<uint64_t, float>* grad_cat,
                 const std::vector<float>& data,
                 const std::vector<uint64_t>& indexes,
                 float label,
                 const std::vector<float>& weight,
                 const std::vector<float>& weight_cat);
//    float GetLoss(const std::vector<std::vector<float>>& data,
//                  const std::vector<float>& label,
//                  const std::vector<float>& weight);
    
private:
    float GetMargin(const std::vector<float>& data,
                    const std::vector<uint64_t>& indexes,
                    float label,
                    const std::vector<float>& weight,
                    const std::vector<float>& weight_cat);
    float Sigma(float x);
};



#include "loss_functions.cpp"
#endif // loss_functions_hpp









//#ifndef loss_functions_hpp
//#define loss_functions_hpp
//
//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <numeric>
//#include <cmath>
//
//#include "printer.hpp"
//#include "data_reader.hpp"
//
//class LossFunction {
//public:
//    virtual std::vector<float> GetGrad(const std::vector<float>& data, float label,
//                                       const std::vector<float>& weight) = 0;
////    virtual std::vector<float> GetBatchGrad(const Batch* batch,
////                                            const std::vector<float>& weight) = 0;
////    virtual float GetLossGetLoss(const std::vector<std::vector<float>> & data, const std::vector<float>& label,
////                                 const std::vector<float>& weight) = 0;
//};
//
//class MSE : public LossFunction {
//public:
//    std::vector<float> GetGrad(const std::vector<float>& data, float label,
//                               const std::vector<float>& weight);
////    std::vector<float> GetBatchGrad(const Batch* batch,
////                                    const std::vector<float>& weight);
////    float GetLoss(const std::vector<std::vector<float>> & data, const std::vector<float>& label,
////                  const std::vector<float>& weight);
//};
//
////class Logistic : public LossFunction {
////public:
////    std::vector<float> GetGrad(const std::vector<float>& data, float label,
////                               const std::vector<float>& weight);
////    std::vector<float> GetBatchGrad(const Batch* batch,
////                                    const std::vector<float>& weight);
////    float GetLoss(const std::vector<std::vector<float>>& data, const std::vector<float>& label,
////                  const std::vector<float>& weight);
////
////private:
////    float GetMargin(const std::vector<float>& data, float label,
////                    const std::vector<float>& weight);
////    float Sigma(float x);
////};
//
//
//
//#include "loss_functions.cpp"
//#endif // loss_functions_hpp
