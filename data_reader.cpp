#include <iostream>
#include <stdlib.h>
#include <utility>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "murmur_hash3.h"
#include "data_reader.h"

DataReader::DataReader(std::string filename, std::string config, uint64_t batchsize, 
                    uint64_t bit_prec, std::string delim, bool generate_names) {
    file_name = filename;
    config_path = config;
    batch_size = batchsize;
    bit_precision = bit_prec;
    delimeter = delim;
    current_index = 0;
    chrs = "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    generator.seed(std::random_device().operator()());
    pick = std::uniform_int_distribution<size_t>(0, chrs.size() - 1);
    length_distribution = std::uniform_int_distribution<uint64_t>(5, 15);
    cat_indices.resize(0);
    if (config_path.size()) {
        ReadConfig();
        if (generate_names) {
            GenerateCatFeatureNames();
        }
    }
    std::ifstream file(filename);
    std::string line = "";
    getline(file, line);
    std::vector<std::string> row_str;
    boost::algorithm::split(row_str, line, boost::is_any_of(delimeter));
    n_features = row_str.size() - 1;
    file.close();

    ComputeNumericIndices();

    input_file.open(filename, std::ifstream::in);
    buffer_size = 1000000;
    if (buffer_size < batch_size) {
        buffer_size = batch_size;
    }
    end_of_data_flag = true;
    available_size = 0;
    data.resize(buffer_size, std::vector<float>(numeric_indices.size()));
    data_cat.resize(buffer_size, std::vector<uint64_t>(cat_indices.size()));
    target.resize(buffer_size);
}

void DataReader::ReadConfig() {
    std::ifstream file(config_path);
    std::string line = "";
    getline(file, line);
    std::vector<std::string> row_str;
    boost::algorithm::split(row_str, line, boost::is_any_of(","));
    if (row_str.size() == 0) {
        printf("error while parsing config file:\n");
        printf("first line should contain indices of cat features, splitted by commas\n");
        exit(1);
    }
    cat_indices.resize(row_str.size());
    std::transform(row_str.begin(), row_str.end(),
                   cat_indices.begin(), [](const std::string& val){return stoi(val);});
    std::sort(cat_indices.begin(), cat_indices.end());
    file.close();
}

void DataReader::ComputeNumericIndices() {
    numeric_indices.reserve(n_features - cat_indices.size());
    size_t index = 0;
    for (size_t i = 1; i < n_features + 1; ++i) {
        if (cat_indices.size() <= index) {
            numeric_indices.push_back(i);
        } else if (cat_indices[index] == i) {
            index++;
        } else {
            numeric_indices.push_back(i);
        }
    }
}

void DataReader::InitFeatureNames(std::vector<std::string> &names) {
    cat_feature_names.resize(names.size());
    std::copy(names.begin(), names.end(), cat_feature_names.begin());
}

std::string DataReader::GenerateRandomString(size_t length)
{
    std::string s;
    s.reserve(length);
    while(length--)
        s += chrs[pick(generator)];
    return s;
}

void DataReader::GenerateCatFeatureNames() {
    uint32_t len;
    cat_feature_names.reserve(cat_indices.size());
    for (size_t i = 0; i < cat_indices.size(); ++i) {
        len = length_distribution(generator);
        cat_feature_names.push_back(GenerateRandomString(len));
    }
}

uint64_t DataReader::GetNumericFeaturesNumber() {
    return n_features - cat_indices.size();
}

uint64_t DataReader::GetCatFeaturesNumber() {
    return cat_indices.size();
}

std::vector<std::string> DataReader::GetFeatureNames() {
    return cat_feature_names;
}

uint64_t DataReader::Hash(std::string s, uint64_t seed) {
    size_t ret = 0;
    auto p = s.begin();
    while (p != s.end())
        if (*p >= '0' && *p <= '9')
            ret = 10*ret + *(p++) - '0';
        else
            return Murmur3(s, seed);
    return ret + seed;
}

uint64_t DataReader::Murmur3(std::string s, uint64_t seed) {
    return MurmurHash3_x64_128(s.c_str(), s.size(), seed);
}

void DataReader::GetData() {
    std::string line = "";
    uint64_t hash;
    bool end_of_file = false;
    for (uint64_t data_index=0; data_index < buffer_size; ++data_index) {
        getline(input_file, line);
        std::vector<std::string> row_str;
        boost::algorithm::split(row_str, line, boost::is_any_of(delimeter));

        if (row_str.size() == 0) {
            end_of_file = true;
            break;
        }
        if (row_str[0].size() == 0) {
            end_of_file = true;
            break;
        }
        for (size_t i = 0; i < numeric_indices.size(); ++i) {
            data[data_index][i] = stof(row_str[numeric_indices[i]]);
        }
        for (size_t i = 0; i < cat_indices.size(); ++i) {
            hash = Hash(cat_feature_names[i], 0);
            hash = Hash(row_str[cat_indices[i]], hash);
            data_cat[data_index][i] = hash % (1 << bit_precision);
        }
        target[data_index] = stof(row_str[0]);
        available_size += 1;

        if (input_file.eof()) {
            end_of_file = true;
            break;
        } else if (!input_file.good()) {
            printf("something wrong happened with reading file\n");
            exit(1);
        }
    }
    if (end_of_file) {
        end_of_file_flag = true;
        input_file.clear();
        input_file.seekg(0);
    }
}

Batch* DataReader::GetBatch() {
    if (end_of_data_flag) {
        available_size = 0;
        while (available_size == 0) {
            GetData();
        }
        current_index = 0;
        end_of_data_flag = false;
    }

    uint64_t start_index, end_index;
    Batch* batch = new Batch;

    start_index = current_index;
    end_index = current_index + batch_size;
    if (end_index >= available_size) {
        end_index = available_size;
        end_of_data_flag = true;
        if (end_of_file_flag) {
            end_of_file_flag = false;
            batch->last_batch = true;
        } else {
            batch->last_batch = false;
        }
    } else {
        current_index = end_index;
        batch->last_batch = false;
    }
    batch->data.resize(end_index - start_index);
    if (numeric_indices.size()) {
        std::copy(data.begin() + start_index, 
                            data.begin() + end_index, batch->data.begin());
    }
    batch->data_cat.resize(end_index - start_index);
    if (cat_indices.size()) {
        std::copy(data_cat.begin() + start_index, 
                        data_cat.begin() + end_index, batch->data_cat.begin());
    }

    batch->target.resize(end_index - start_index);
    std::copy(target.begin() + start_index, 
            target.begin() + end_index, batch->target.begin());
    return batch;
}

BatchIterator* DataReader::GetBatchIterator() {
    if (end_of_data_flag) {
        available_size = 0;
        while (available_size == 0) {
            GetData();
        }
        current_index = 0;
        end_of_data_flag = false;
    }

    uint64_t start_index, end_index;
    BatchIterator* batch_iter = new BatchIterator;

    start_index = current_index;
    end_index = current_index + batch_size;
    if (end_index >= available_size) {
        end_index = available_size;
        end_of_data_flag = true;
        if (end_of_file_flag) {
            end_of_file_flag = false;
            batch_iter->last_batch = true;
        } else {
            batch_iter->last_batch = false;
        }
    } else {
        current_index = end_index;
        batch_iter->last_batch = false;
    }

    batch_iter->size = end_index - start_index;

    batch_iter->data_it = data.begin() + start_index;
    batch_iter->data_cat_it = data_cat.begin() + start_index;
    batch_iter->target_it = target.begin() + start_index;
    return batch_iter;
}

void PrintBatch(Batch* batch) {
    int len = batch->data.size();
    for (int i = 0; i < len; ++i) {
        std::cout << "target: " << batch->target[i] << ";\t";
        for (auto data_it = batch->data[i].begin();
             data_it != batch->data[i].end(); ++data_it) {
            std::cout << *data_it << " ";
        }
        std::cout << ";\t";
        std::cout << "cat:";
        for (auto data_it = batch->data_cat[i].begin();
             data_it != batch->data_cat[i].end(); ++data_it) {
            std::cout << *data_it << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void PrintBatchInFile(Batch* batch, std::string output_file) {
    std::ofstream fout(output_file, std::ios::app);

    int len = batch->data.size();
    for (int i = 0; i < len; ++i) {
        fout << "target: " << batch->target[i] << ";\t";
        for (auto data_it = batch->data[i].begin();
             data_it != batch->data[i].end(); ++data_it) {
            fout << *data_it << " ";
        }
        fout << ";\t";
        fout << "cat:";
        for (auto data_it = batch->data_cat[i].begin();
             data_it != batch->data_cat[i].end(); ++data_it) {
            fout << *data_it << " ";
        }
        fout << std::endl;
    }
    fout << std::endl;
    fout << std::endl;
}
