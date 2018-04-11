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
    const char *loss = "MSE";
    float lr = 0.000001;
    int iterations = 5000000;
    const char *optimizer = "SGD";
};

struct ApplyArgs_t {
    const char *input_path;
    const char *output_path;
    const char *model_path = "model.bin";
    const char *delimiter = ",";
    const char *loss;
    float lr;
    int iterations;
    const char *optimizer;
};

FitArgs_t ParseFitParameters(int argc, char* argv[]);
ApplyArgs_t ParseApplyParameters(int argc, char* argv[]);

#include "argument_parser.cpp"
#endif // argument_parser_hpp
