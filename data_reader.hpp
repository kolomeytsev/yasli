#ifndef data_reader_hpp
#define data_reader_hpp

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <string>
#include <boost/algorithm/string.hpp>

class DataReader {
public:
    DataReader(std::string filename, std::string delim = ",") :
    file_name(filename), delimeter(delim) {}
    void GetData();
    std::pair<std::vector<float>, int> GetRow();
    
private:
    std::string file_name;
    std::string delimeter;
    
    std::vector<std::vector<float> > data;
    std::vector<int> target;
};

void PrintRow(std::pair<std::vector<float>, int> row_data);

#include "data_reader.cpp"
#endif // data_reader_hpp
