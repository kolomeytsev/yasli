#include "data_reader.hpp"

#ifndef data_reader_cpp
#define data_reader_cpp

void DataReader::GetData()
{
    std::ifstream file(file_name);
    //std::vector<std::vector<std::string> > data;
    std::string line = "";
    while (getline(file, line)) {
        std::vector<std::string> row_str;
        boost::algorithm::split(row_str, line, boost::is_any_of(delimeter));
        std::vector<float> row_float(row_str.size());
        std::transform(row_str.begin(), row_str.end(),
                                     row_float.begin(), [](const std::string& val){return stof(val);});
        std::vector<float> features(row_float.begin() + 1, row_float.end());
        data.push_back(features);
        target.push_back(row_float[0]);
    }
    file.close();
}

std::pair<std::vector<float>, int> DataReader::GetRow()
{
    int random_index = rand() % data.size();
    return std::pair<std::vector<float>, int>(data[random_index], target[random_index]);
}

void PrintRow(std::pair<std::vector<float>, int> row_data)
{
    std::cout << "target: " << row_data.second << " ";
    for (std::vector<float>::iterator it = row_data.first.begin();
             it != row_data.first.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

#endif // data_reader_cpp
