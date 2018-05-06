#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

#include "loss_functions.hpp"
#include "optimizers.hpp"
#include "printer.hpp"
#include "data_reader.hpp"
#include "argument_parser.hpp"


class Model {
public:
    Model(FitArgs_t fit_args, DataReader* data_reader);
    void Fit();
    void Save();
    void InitWeights(std::vector<float>& weights);
    void Predict(std::string output_path);
private:
    FitArgs_t fit_args;
    std::vector<float> weights;
    DataReader* data_reader;
    LossFunction* loss_function;
    Optimizer* optimizer;
};

Model::Model(FitArgs_t fit_args, DataReader* data_reader) : 
            fit_args(fit_args), data_reader(data_reader) {
    if (fit_args.loss == "mse") {
        loss_function = new MSE();
    } else if (fit_args.loss == "logistic") {
            loss_function = new Logistic();
    } else {
        printf("unknown loss function\n");
        std::cout << fit_args.loss << std::endl;
        exit(1);
    }
    if (fit_args.optimizer == "sgd") {
        optimizer = new SGD(fit_args.lr);
    } else if (fit_args.optimizer == "adagrad") {
        optimizer = new Adagrad(data_reader->GetRow().first.size(),
                                fit_args.lr);
    } else {
        printf("unknown optimizer\n");
        exit(1);
    }
}

void Model::Fit() {
    weights.resize(data_reader->GetRow().first.size());
    for (int index = 0; index < 100; ++index) {
        std::pair<std::vector<float>, int> row = data_reader->GetRow();
        std::vector<float> grad = loss_function->GetGrad(row, weights);
        weights = optimizer->UpdateWeight(weights, grad);
    }
    PrintVector(weights);
    std::cout << std::endl;
}

void Model::Save() {
    std::ofstream fout(fit_args.model_path);
    fout << fit_args.loss <<  " ";
    fout << fit_args.optimizer <<  " ";
    fout << fit_args.lr << std::endl;
    for (auto it=weights.begin(); it!=weights.end() - 1; ++it) {
        fout << *it <<  " ";
    }
    fout << *(weights.end() - 1) << std::endl;
    fout.close();
}

void Model::InitWeights(std::vector<float>& new_weights) {
    weights.resize(new_weights.size());
    std::copy(new_weights.begin(), new_weights.end(), weights.begin());
}

void Model::Predict(std::string output_path) {
    printf("Predicting\n");
    std::vector<float> prediction;

    // Paste the code here

    prediction.push_back(0.66);
    prediction.push_back(0.33);
    prediction.push_back(0.22);

    std::ofstream fout(output_path);
    for (auto it=prediction.begin(); it!=prediction.end(); ++it) {
        fout << *it <<  std::endl;
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

void LoadModel(std::vector<float> &weights, FitArgs_t &fit_args, 
                std::string model_path) {
    std::ifstream fin(model_path);
    fin >> fit_args.loss;
    fin >> fit_args.optimizer;
    fin >> fit_args.lr;
    std::cout << fit_args.lr << std::endl;

    std::vector<float> loaded_weights;
    float weight;
    fin >> weight;
    while (!fin.eof()){
        loaded_weights.push_back(weight);
        fin >> weight;
    }
    fin.close();
    weights.resize(loaded_weights.size());
    std::copy(loaded_weights.begin(), loaded_weights.end(), weights.begin());
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
    FitArgs_t fit_args;
    fit_args = ParseFitParameters(argc, argv);

    DataReader reader(fit_args.input_path, fit_args.batch_size, 
                        fit_args.delimiter);
    reader.GetData();
    Model model(fit_args, &reader);
    model.Fit();
    model.Save();
}

void Apply(int argc, char* argv[]) {
    ApplyArgs_t apply_args;
    FitArgs_t fit_args;
    std::vector<float> weights;

    apply_args = ParseApplyParameters(argc, argv);
    DataReader reader(apply_args.input_path, apply_args.batch_size, 
                        apply_args.delimiter);
    LoadModel(weights, fit_args, apply_args.model_path);
    Model model(fit_args, &reader);
    model.InitWeights(weights);

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
    std::cout << argv[1] << std::endl;
    if (strcmp(argv[1], "fit") == 0) {
        Fit(argc, argv);
    } else {
        if (strcmp(argv[1], "apply") == 0) {
            Apply(argc, argv);
        } else {
            printf("found unknown mode option: only \"fit\" and \"apply\" modes are available.\n");
        }
    }
    return 0;
}
