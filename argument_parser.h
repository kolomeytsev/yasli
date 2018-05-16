#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include <string>

struct FitArgs {
    std::string input_path;
    std::string model_path = "model.bin";
    std::string delimiter = ",";
    std::string loss = "mse";
    std::string optimizer = "sgd";
    std::string config_path = "";
    float lr = 0.1;
    int epochs = 100;
    int batch_size = 64;
    uint64_t bit_precision = 18;
    float ftrl_alpha = 0.005;
    float ftrl_beta = 0.1;
    float l1 = 0;
    float l2 = 0;
};

struct ApplyArgs {
    std::string input_path;
    std::string output_path;
    std::string config_path = "";
    std::string model_path = "model.bin";
    std::string delimiter = ",";
    int batch_size = 64;
};

FitArgs ParseFitParameters(int argc, char* argv[]);
ApplyArgs ParseApplyParameters(int argc, char* argv[]);

#endif // ARGUMENT_PARSER_H
