#ifndef argument_parser_hpp
#define argument_parser_hpp

#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <getopt.h>


struct FitArgs_t {
    std::string input_path;
    std::string model_path = "model.bin";
    std::string delimiter = ",";
    std::string loss = "mse";
    std::string optimizer = "sgd";
    float lr = 0.000001;
    int epochs = 100;
    int batch_size = 64;
};

struct ApplyArgs_t {
    std::string input_path;
    std::string output_path;
    std::string model_path = "model.bin";
    std::string delimiter = ",";
    int batch_size = 64;
};

FitArgs_t ParseFitParameters(int argc, char* argv[]);
ApplyArgs_t ParseApplyParameters(int argc, char* argv[]);

#include "argument_parser.cpp"
#endif // argument_parser_hpp
