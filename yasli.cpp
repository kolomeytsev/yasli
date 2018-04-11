#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include "loss_functions.hpp"
#include "optimizers.hpp"
#include "printer.hpp"
#include "data_reader.hpp"
#include "argument_parser.hpp"


class Model {
public:
    std::vector<float> Fit();
    Model(LossFunction* loss_function, Optimizer* optimizer, 
        DataReader* data_reader, int max_iter) :
        loss_function(loss_function), optimizer(optimizer), 
        data_reader(data_reader), max_iter(max_iter) {};
private:
    DataReader* data_reader;
    LossFunction* loss_function;
    Optimizer* optimizer;
    int max_iter;
};

std::vector<float> Model::Fit() {
    std::vector<float> weight;
    weight.resize(data_reader->GetRow().first.size());
    for (int index = 0; index < max_iter; ++index) {
        std::pair<std::vector<float>, int> row = data_reader->GetRow();
        std::vector<float> grad = loss_function->GetGrad(row, weight);
        weight = optimizer->UpdateWeight(weight, grad);
    }
    PrintVector(weight);
    std::cout << std::endl;
    return weight;
}

void Fit(int argc, char* argv[]) {
    FitArgs_t fit_args;
    fit_args = ParseFitParameters(argc, argv);

    DataReader reader(fit_args.input_path, fit_args.delimiter);
    reader.GetData();
    LossFunction* loss_function;
    Optimizer* optimizer;
    if (strcmp(fit_args.loss, "mse") == 0) {
        loss_function = new MSE();
    } else {
        if (strcmp(fit_args.loss, "logistic") == 0) {
            loss_function = new Logistic();
        } else {
            printf("unknown loss function\n");
            exit(1);
        }
    }
    if (strcmp(fit_args.optimizer, "sgd") == 0) {
        optimizer = new SGD(fit_args.lr);
    } else {
        if (strcmp(fit_args.optimizer, "adagrad") == 0) {
            optimizer = new Adagrad(reader.GetRow().first.size(), fit_args.lr);
        } else {
            printf("unknown optimizer\n");
            exit(1);
        }
    }
    Model model(loss_function, optimizer, &reader, fit_args.iterations);
    model.Fit();
}

void Apply(int argc, char* argv[]) {
    ApplyArgs_t apply_args;
    apply_args = ParseApplyParameters(argc, argv);
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
