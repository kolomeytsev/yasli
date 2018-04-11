#include "printer.hpp"


#ifndef printer_cpp
#define printer_cpp

std::vector< std::pair<std::vector<float>, float> > ReadData() {
    std::ifstream file("test_data.txt");
    
    std::vector< std::pair<std::vector<float>, float> > data;
    int count;
    file >> count;
    data.reserve(count);
    
    int len;
    file >> len;
    
    std::cout << count << ' ' << len << std::endl;
    
    for (int index = 0; index < count; ++index) {
        std::vector<float> X;
        X.reserve(len);
        for (int second_index = 0; second_index < len; ++second_index) {
            float tmp;
            file >> tmp;
            X.push_back(tmp);
        }
        float label;
        file >> label;
        data.push_back({X, label});
    }
    return data;
}

void PrintVector(const std::vector<float>& vec) {
    for (auto val : vec) {
        std::cout << ' ' << val;
    }
}

void PrintMatrixconst (const std::vector<std::vector<float>>& mat) {
    for (const auto& vec : mat) {
        PrintVector(vec);
    }
    std::cout << std::endl;
}

void PrintDataconst (const std::vector< std::pair<std::vector<float>, float> >& data) {
    for (const auto& row : data) {
        PrintVector(row.first);
        std::cout << ' ' << row.second << std::endl;
    }
}

#endif // printer_cpp
