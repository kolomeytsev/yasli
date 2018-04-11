#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>

#include <vector>
#include <random>

#include "loss_functions.hpp"
#include "optimizers.hpp"
#include "printer.hpp"
#include "data_reader.hpp"

struct globalFitArgs_t {
    const char *input_path;
    const char *model_path = "model.bin";
    const char *delimiter = ",";
    const char *loss = "MSE";
    float lr = 0.00001;
    const char *optimizer = "SGD";
} globalFitArgs;

struct globalApplyArgs_t {
    const char *input_path;
    const char *output_path;
    const char *model_path = "model.bin";
    const char *delimiter = ",";
    const char *loss;
    float lr;
    const char *optimizer;
} globalApplyArgs;

void ParseFitParameters(int argc, char* argv[])
{
    const char* short_options = "hi:o:d:l:w:O:";
    const struct option long_options[] = {
        {"help",no_argument,NULL,'h'},
        {"input-path",required_argument,NULL,'i'},
        {"model-path",required_argument,NULL,'m'},
        {"delimiter",required_argument,NULL,'d'},
        {"loss-function",required_argument,NULL,'l'},
        {"learning-rate",required_argument,NULL,'w'},
        {"optimizer",required_argument,NULL,'O'},
        {NULL,0,NULL,0}
    };

    int opt;
    int option_index;

    while ((opt = getopt_long(argc, argv, short_options,
            long_options, &option_index)) != -1){

        switch(opt) {
            case 'h': {
                printf("Help:\n");
                break;
            }
            case 'i': {
                globalFitArgs.input_path = optarg;
                break;
            }
            case 'm': {
                globalFitArgs.model_path = optarg;
                break;
            }
            case 'd': {
                globalFitArgs.delimiter = optarg;
                break;
            }
            case 'l': {
                globalFitArgs.loss = optarg;
                break;
            }
            case 'w': {
                globalFitArgs.lr = std::stof(optarg);
                break;
            }
            case 'O': {
                globalFitArgs.optimizer = optarg;
                break;
            }
            default: {
                printf("found unknown option\n");
                break;
            }
        }
    }
}

void ParseApplyParameters(int argc, char* argv[])
{
    const char* short_options = "hi:o:d:l:w:O:";
    const struct option long_options[] = {
        {"help",no_argument,NULL,'h'},
        {"input-path",required_argument,NULL,'i'},
        {"model-path",required_argument,NULL,'m'},
        {"delimiter",required_argument,NULL,'d'},
        {"loss-function",required_argument,NULL,'l'},
        {"learning-rate",required_argument,NULL,'w'},
        {"optimizer",required_argument,NULL,'O'},
        {NULL,0,NULL,0}
    };

    int opt;
    int option_index;

    while ((opt = getopt_long(argc, argv, short_options,
            long_options, &option_index)) != -1){

        switch(opt) {
            case 'h': {
                printf("Help:\n");
                break;
            }
            case 'i': {
                globalApplyArgs.input_path = optarg;
                break;
            }
            case 'o': {
                globalApplyArgs.output_path = optarg;
                break;
            }
            case 'm': {
                globalApplyArgs.model_path = optarg;
                break;
            }
            case 'd': {
                globalApplyArgs.delimiter = optarg;
                break;
            }
            case 'l': {
                globalApplyArgs.loss = optarg;
                break;
            }
            case 'w': {
                globalApplyArgs.lr = std::stof(optarg);
                break;
            }
            case 'O': {
                globalApplyArgs.optimizer = optarg;
                break;
            }
            default: {
                printf("found unknown option\n");
                break;
            }
        }
    }
}

class Model {
public:
    std::vector<float> Fit();
    Model(LossFunction* loss_function, Optimizer* optimizer, DataReader* data_reader) :
        loss_function(loss_function), optimizer(optimizer), data_reader(data_reader) {};
private:
    DataReader* data_reader;
    LossFunction* loss_function;
    Optimizer* optimizer;
};

std::vector<float> Model::Fit() {
    std::vector<float> weight;
    weight.resize(data_reader->GetRow().first.size());
    for (int index = 0; index < 5000000; ++index) {
        std::pair<std::vector<float>, int> row = data_reader->GetRow();
        std::vector<float> grad = loss_function->GetGrad(row, weight);
        weight = optimizer->UpdateWeight(weight, grad);
    }
    PrintVector(weight);
    std::cout << std::endl;
    return weight;
}

void Fit(int argc, char* argv[])
{
    printf("fitting\n");
    ParseFitParameters(argc, argv);
    
    DataReader reader("test_logistic_data.csv");
    reader.GetData();
    Logistic logistic;
    SGD sgd;
    Model model(&logistic, &sgd, &reader);
    model.Fit();
    return 0;
}

void Apply(int argc, char* argv[])
{
    printf("applying\n");
    ParseApplyParameters(argc, argv);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("Usage:\n");
        printf("Training a model:\n");
        printf("yasli fit -f <file path> [optional parameters]\n");
        printf("\nApplying a model:\n");
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