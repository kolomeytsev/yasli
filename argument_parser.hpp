#ifndef argument_parser_hpp
#define argument_parser_hpp

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>


struct FitArgs_t {
    const char *input_path;
    const char *model_path = "model.bin";
    const char *delimiter = ",";
    const char *loss = "mse";
    float lr = 0.000001;
    int epochs = 100;
    int batch_size = 64;
    const char *optimizer = "sgd";
};

struct ApplyArgs_t {
    const char *input_path;
    const char *output_path;
    const char *model_path = "model.bin";
    const char *delimiter = ",";
    const char *loss;
    float lr;
    int batch_size = 64;
    const char *optimizer;
};

FitArgs_t ParseFitParameters(int argc, char* argv[]);
ApplyArgs_t ParseApplyParameters(int argc, char* argv[]);

#include "argument_parser.cpp"
#endif // argument_parser_hpp
