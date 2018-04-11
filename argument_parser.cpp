#include "argument_parser.hpp"

#ifndef argument_parser_cpp
#define argument_parser_cpp

FitArgs_t ParseFitParameters(int argc, char* argv[]) {
    const char* short_options = "hi:o:d:l:w:O:I:";
    const struct option long_options[] = {
        {"help",no_argument,NULL,'h'},
        {"input-path",required_argument,NULL,'i'},
        {"model-path",required_argument,NULL,'m'},
        {"delimiter",required_argument,NULL,'d'},
        {"loss-function",required_argument,NULL,'l'},
        {"optimizer",required_argument,NULL,'O'},
        {"learning-rate",required_argument,NULL,'w'},
        {"iterations",required_argument,NULL,'I'},
        {NULL,0,NULL,0}
    };

    int opt;
    int option_index;
    FitArgs_t fit_args;

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
            case 'I': {
                fit_args.iterations = std::stoi(optarg);
                break;
            }
            default: {
                printf("found unknown option\n");
                break;
            }
        }
    }
    return fit_args;
}

ApplyArgs_t ParseApplyParameters(int argc, char* argv[]) {
    const char* short_options = "hi:o:d:l:w:O:I:";
    const struct option long_options[] = {
        {"help",no_argument,NULL,'h'},
        {"input-path",required_argument,NULL,'i'},
        {"model-path",required_argument,NULL,'m'},
        {"delimiter",required_argument,NULL,'d'},
        {"loss-function",required_argument,NULL,'l'},
        {"optimizer",required_argument,NULL,'O'},
        {"learning-rate",required_argument,NULL,'w'},
        {"iterations",required_argument,NULL,'I'},
        {NULL,0,NULL,0}
    };

    int opt;
    int option_index;
    ApplyArgs_t apply_args;

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
            case 'l': {
                apply_args.loss = optarg;
                break;
            }
            case 'w': {
                apply_args.lr = std::stof(optarg);
                break;
            }
            case 'O': {
                apply_args.optimizer = optarg;
                break;
            }
            default: {
                printf("found unknown option\n");
                break;
            }
        }
    }
    return apply_args;
}

#endif // argument_parser_cpp
