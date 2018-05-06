#include "data_reader.hpp"

#ifndef data_reader_cpp
#define data_reader_cpp

void DataReader::GetData() {
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

std::pair<std::vector<float>, float> DataReader::GetRow() {
    int random_index = rand() % data.size();
    return std::pair<std::vector<float>, float>(data[random_index], target[random_index]);
}

Batch* DataReader::GetBatch() {
    int start_index, end_index;
    Batch* batch = new Batch;

    start_index = current_index;
    end_index = current_index + batch_size;
    if (end_index >= data.size()) {
        end_index = data.size();
        batch->last_batch = true;
        current_index = 0;
    } else {
        batch->last_batch = false;
        current_index = end_index;
    }
    batch->data = std::vector<std::vector<float> >(end_index - start_index);
    std::copy(data.begin() + start_index, 
            data.begin() + end_index, batch->data.begin());
    batch->target = std::vector<float>(end_index - start_index);
    std::copy(target.begin() + start_index, 
            target.begin() + end_index, batch->target.begin());
    return batch;
}

void PrintRow(std::pair<std::vector<float>, float> row_data) {
    std::cout << "target: " << row_data.second << " ";
    for (std::vector<float>::iterator it = row_data.first.begin();
             it != row_data.first.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

void PrintBatch(Batch* batch) {
    int len = batch->data.size();
    for (int i = 0; i < len; ++i) {
        std::cout << "target: " << batch->target[i] << " ";
        for (auto data_it = batch->data[i].begin();
             data_it != batch->data[i].end(); ++data_it) {
            std::cout << *data_it << " ";
        }
        std::cout << std::endl;
    }
}

#endif // data_reader_cpp
