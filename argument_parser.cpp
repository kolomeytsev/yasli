#include <stdlib.h>
#include <getopt.h>
#include <stdio.h>

#include "argument_parser.h"

FitArgs ParseFitParameters(int argc, char* argv[]) {
    const char* short_options = "hi:o:m:d:l:w:O:e:B:b:c:0:1:2:3:";
    const struct option long_options[] = {
        {"help",no_argument,NULL,'h'},
        {"input-path",required_argument,NULL,'i'},
        {"model-path",required_argument,NULL,'m'},
        {"delimiter",required_argument,NULL,'d'},
        {"loss-function",required_argument,NULL,'l'},
        {"optimizer",required_argument,NULL,'O'},
        {"learning-rate",required_argument,NULL,'w'},
        {"epochs",required_argument,NULL,'e'},
        {"batch",required_argument,NULL,'B'},
        {"bits",required_argument,NULL,'b'},
        {"config",required_argument,NULL,'c'},
        {"ftrl_alpha",required_argument,NULL,0},
        {"ftrl_beta",required_argument,NULL,1},
        {"l1",required_argument,NULL,2},
        {"l2",required_argument,NULL,3},
        {NULL,0,NULL,0}
    };
    
    int opt;
    int option_index;
    FitArgs fit_args;
    
    while ((opt = getopt_long(argc, argv, short_options,
                              long_options, &option_index)) != -1){
        
        switch(opt) {
            case 'h': {
                printf("Help:\n");
                break;
            }
            case 'i': {
                fit_args.input_path = optarg;
                break;
            }
            case 'm': {
                fit_args.model_path = optarg;
                break;
            }
            case 'd': {
                fit_args.delimiter = optarg;
                break;
            }
            case 'l': {
                fit_args.loss = optarg;
                break;
            }
            case 'w': {
                fit_args.lr = std::stof(optarg);
                break;
            }
            case 'O': {
                fit_args.optimizer = optarg;
                break;
            }
            case 'e': {
                fit_args.epochs = std::stoi(optarg);
                break;
            }
            case 'B': {
                fit_args.batch_size = std::stoi(optarg);
                break;
            }
            case 'b': {
                fit_args.bit_precision = std::stoi(optarg);
                break;
            }
            case 'c': {
                fit_args.config_path = optarg;
                break;
            }
            case 0: {
                fit_args.ftrl_alpha = std::stof(optarg);
                break;
            }
            case 1: {
                fit_args.ftrl_beta = std::stof(optarg);
                break;
            }
            case 2: {
                fit_args.l1 = std::stof(optarg);
                break;
            }
            case 3: {
                fit_args.l2 = std::stof(optarg);
                break;
            }
            default: {
                printf("found unknown option\n");
                exit(1);
            }
        }
    }
    return fit_args;
}

ApplyArgs ParseApplyParameters(int argc, char* argv[]) {
    const char* short_options = "hi:o:m:d:B:c:";
    const struct option long_options[] = {
        {"help",no_argument,NULL,'h'},
        {"input-path",required_argument,NULL,'i'},
        {"output-path",required_argument,NULL,'o'},
        {"model-path",required_argument,NULL,'m'},
        {"delimiter",required_argument,NULL,'d'},
        {"batch",required_argument,NULL,'B'},
        {"config",required_argument,NULL,'c'},
        {NULL,0,NULL,0}
    };
    
    int opt;
    int option_index;
    ApplyArgs apply_args;
    
    while ((opt = getopt_long(argc, argv, short_options,
                              long_options, &option_index)) != -1){
        
        switch(opt) {
            case 'h': {
                printf("Help:\n");
                break;
            }
            case 'i': {
                apply_args.input_path = optarg;
                break;
            }
            case 'o': {
                apply_args.output_path = optarg;
                break;
            }
            case 'm': {
                apply_args.model_path = optarg;
                break;
            }
            case 'd': {
                apply_args.delimiter = optarg;
                break;
            }
            case 'B': {
                apply_args.batch_size = std::stoi(optarg);
                break;
            }
            case 'c': {
                apply_args.config_path = optarg;
                break;
            }
            default: {
                printf("found unknown option\n");
                exit(1);
            }
        }
    }
    return apply_args;
}
