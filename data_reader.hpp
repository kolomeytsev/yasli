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

struct Batch {
    std::vector<std::vector<float> > data;
    std::vector<float> target;
    bool last_batch;
};

class DataReader {
public:
    DataReader(std::string filename, int batch_size, std::string delim = ",") :
                file_name(filename), batch_size(batch_size), 
                delimeter(delim), current_index(0) {}
    void GetData();
    std::pair<std::vector<float>, float> GetRow();
    Batch* GetBatch();
    
private:
    std::string file_name;
    std::string delimeter;
    
    std::vector<std::vector<float> > data;
    std::vector<float> target;

    int current_index;
    int batch_size;
};

void PrintRow(std::pair<std::vector<float>, float> row_data);
void PrintBatch(Batch* batch);

#include "data_reader.cpp"
#endif // data_reader_hpp
