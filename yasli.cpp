#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <string.h>

#include "loss_functions.h"
#include "optimizers.h"
#include "data_reader.h"
#include "argument_parser.h"

class Model {
public:
    Model(FitArgs fit_args, DataReader* data_reader);
    ~Model();
    void Fit();
    void Save();
    void InitWeights(std::vector<float>& new_weights, 
                    std::vector<float> &new_weights_cat);
    void Predict(std::string output_path);
private:
    FitArgs fit_args;
    std::vector<float> weights;
    std::vector<float> weights_cat;
    DataReader* data_reader;
    LossFunction* loss_function;
    Optimizer* optimizer;
    
    std::vector<float> PredictBatch(BatchIterator *batch_iter);
};

void TestBatchIteration(BatchIterator* batch_iter);

Model::Model(FitArgs fit_args, DataReader* data_reader) :
fit_args(fit_args), data_reader(data_reader) {
    weights.resize(data_reader->GetNumericFeaturesNumber());
    weights_cat.resize(1 << fit_args.bit_precision);
    if (fit_args.loss == "mse") {
        loss_function = new MSE();
    } else if (fit_args.loss == "logistic") {
        loss_function = new Logistic();
    } else {
        std::cout << "unknown loss function" << std::endl;
        std::cout << fit_args.loss << std::endl;
        exit(1);
    }
    if (fit_args.optimizer == "sgd") {
        optimizer = new SGD(fit_args.lr);
    } else if (fit_args.optimizer == "adagrad") {
        optimizer = new Adagrad(data_reader->GetNumericFeaturesNumber() +
                                (1 << fit_args.bit_precision), fit_args.lr);
    } else if (fit_args.optimizer == "ftrl") {
        if (fit_args.loss != "logistic") {
            std::cout << "bad loss function: for ftrl method use logistic";
            std::cout << std::endl;
            exit(1);
        }
        optimizer = new Ftrl(fit_args.lr, fit_args.ftrl_alpha, fit_args.ftrl_beta,
                                fit_args.l1, fit_args.l2,
                                data_reader->GetNumericFeaturesNumber(),
                                1 << fit_args.bit_precision);
    } else {
        printf("unknown optimizer\n");
        exit(1);
    }
}

Model::~Model() {
    delete loss_function;
    delete optimizer;
}

void Model::Fit() {
    std::cout << "num epoch = " << fit_args.epochs << std::endl;
    std::vector<float> grad;
    grad.resize(weights.size());
    for (int index = 0; index < fit_args.epochs; ++index) {
        bool last_batch = false;
        while (!last_batch) {
            BatchIterator* batch_iter = data_reader->GetBatchIterator();
            last_batch = batch_iter->last_batch;
            if (fit_args.optimizer == "ftrl") {
                optimizer->UpdateWeightFtrl(&weights, &weights_cat, batch_iter);
            } else {
                std::fill(grad.begin(), grad.end(), 0);
                std::unordered_map<uint64_t, float> grad_cat;
                loss_function->GetBatchGrad(&grad, &grad_cat, batch_iter, weights, weights_cat);
                optimizer->UpdateWeight(&weights, &weights_cat, grad, grad_cat);
            }
            delete batch_iter;
        }
    }
}

void Model::Save() {
    std::ofstream fout(fit_args.model_path);
    fout << fit_args.loss <<  " ";
    fout << fit_args.optimizer <<  " ";
    fout << fit_args.lr << " ";
    fout << fit_args.bit_precision << std::endl;

    fout << weights.size() << " ";
    if (data_reader->GetNumericFeaturesNumber()) {
        for (auto it=weights.begin(); it!=weights.end() - 1; ++it) {
            fout << *it <<  " ";
        }
        fout << *(weights.end() - 1);
    }
    fout << std::endl;

    fout << weights_cat.size() << " ";
    if (data_reader->GetCatFeaturesNumber()) {
        for (auto it=weights_cat.begin(); it!=weights_cat.end() - 1; ++it) {
            fout << *it <<  " ";
        }
        fout << *(weights_cat.end() - 1);
    }
    fout << std::endl;
    if (data_reader->GetCatFeaturesNumber()) {
        std::vector<std::string> feature_names = data_reader->GetFeatureNames();
        fout << feature_names.size() << " ";
        for (auto it=feature_names.begin(); it!=feature_names.end() - 1; ++it) {
            fout << *it <<  " ";
        }
        fout << *(feature_names.end() - 1);
    }
    fout << std::endl;
    fout.close();
}

void Model::InitWeights(std::vector<float>& new_weights,
                        std::vector<float> &new_weights_cat) {
    weights.resize(new_weights.size());
    std::copy(new_weights.begin(), new_weights.end(), weights.begin());
    weights_cat.resize(new_weights_cat.size());
    std::copy(new_weights_cat.begin(), new_weights_cat.end(), weights_cat.begin());
}

std::vector<float> Model::PredictBatch(BatchIterator *batch_iter) {
    uint64_t len = batch_iter->size;
    std::vector<float> predictions;
    predictions.reserve(len);
    auto data_cat_it = batch_iter->data_cat_it;
    for (auto it = batch_iter->data_it; it != batch_iter->data_it + len; ++it) {
        float prediction = std::inner_product(weights.begin(), weights.end(), it->begin(), 0.0);
        for (auto it_inner=data_cat_it->begin(); it_inner != data_cat_it->end(); ++it_inner) {
            prediction += weights_cat[*it_inner];
        }
        ++data_cat_it;
        predictions.push_back(prediction);
    }
    return predictions;
}

void Model::Predict(std::string output_path) {
    printf("Predicting\n");
    bool done = false;
    std::ofstream fout(output_path);
    while (!done) {
        BatchIterator* batch_iter = data_reader->GetBatchIterator();
        done = batch_iter->last_batch;
        std::vector<float> batch_prediction = PredictBatch(batch_iter);
        for (auto it=batch_prediction.begin(); it!=batch_prediction.end(); ++it) {
            fout << *it <<  std::endl;
        }
        delete batch_iter;
    }
    fout.close();
}

void TestBatch(DataReader* dr, int epochs) {
    DataReader* data_reader = dr;
    Batch* batch;
    bool done;
    int index;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        done = false;
        index = 1;
        printf("epoch number: %d\n", epoch + 1);
        while (!done) {
            batch = data_reader->GetBatch();
            done = batch->last_batch;
            printf("batch number: %d\n", index);
            PrintBatch(batch);
            index++;
        }
        printf("epoch ended\n");
    }
}

void TestBatchIteration(BatchIterator* batch_iter) {
    std::cout << "testing iteration" << std::endl;
    for (auto it = batch_iter->data_it; it != batch_iter->data_it + batch_iter->size; ++it) {
        for (auto it_data=it->begin(); it_data != it->end(); ++it_data) {
            std::cout << *it_data << " ";
        }
        std::cout << std::endl;
    }
}

void LoadModel(std::vector<float> &weights, std::vector<float> &weights_cat, 
               std::vector<std::string> &feature_names,
               FitArgs &fit_args, std::string model_path) {
    std::ifstream fin(model_path);
    fin >> fit_args.loss;
    fin >> fit_args.optimizer;
    fin >> fit_args.lr;
    fin >> fit_args.bit_precision;
    
    uint64_t size;
    fin >> size;
    weights.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
        fin >> weights[i];
    }
    fin >> size;
    weights_cat.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
        fin >> weights_cat[i];
    }
    if (size > 0) {
        fin >> size;
        feature_names.resize(size);
        for (uint64_t i = 0; i < size; ++i) {
            fin >> feature_names[i];
        }
    }
    fin.close();
}

void SavePrediction(std::vector<float> &predictions,
                    std::string output_path) {
    std::ofstream fout(output_path);
    for (auto it=predictions.begin(); it!=predictions.end(); ++it) {
        fout << *it <<  std::endl;
    }
    fout.close();
}

void Fit(int argc, char* argv[]) {
    FitArgs fit_args;
    fit_args = ParseFitParameters(argc, argv);
    DataReader reader(fit_args.input_path, fit_args.config_path,
                      fit_args.batch_size, fit_args.bit_precision, fit_args.delimiter);
    Model model(fit_args, &reader);
    model.Fit();
    model.Save();
}

void Apply(int argc, char* argv[]) {
    ApplyArgs apply_args;
    FitArgs fit_args;
    std::vector<float> weights;
    std::vector<float> weights_cat;
    std::vector<std::string> feature_names;
    
    apply_args = ParseApplyParameters(argc, argv);
    LoadModel(weights, weights_cat, feature_names, 
                fit_args, apply_args.model_path);
    
    DataReader reader(apply_args.input_path, apply_args.config_path, apply_args.batch_size, 
                            fit_args.bit_precision, apply_args.delimiter, false);
    if (feature_names.size()) {
        reader.InitFeatureNames(feature_names);
    }
    Model model(fit_args, &reader);
    model.InitWeights(weights, weights_cat);
    model.Predict(apply_args.output_path);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("Training a model:\n");
        printf("yasli fit -f <file path> [optional parameters]\n");
        printf("Applying a model:\n");
        printf("yasli apply [optional parameters]\n");
    }
    if (strcmp(argv[1], "fit") == 0) {
        Fit(argc, argv);
    } else if (strcmp(argv[1], "apply") == 0) {
        Apply(argc, argv);
    } else {
        printf("found unknown mode option: only \"fit\" and \"apply\" modes are available.\n");
    }
    return 0;
}
