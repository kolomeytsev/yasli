#ifndef DATA_READER_H
#define DATA_READER_H

#include <fstream>
#include <vector>
#include <random>
#include <string>

struct Batch {
    std::vector<std::vector<float>> data;
    std::vector<std::vector<uint64_t>> data_cat;
    std::vector<float> target;
    bool last_batch;
};

struct BatchIterator {
    std::vector<std::vector<float>>::const_iterator data_it;
    std::vector<std::vector<uint64_t>>::const_iterator data_cat_it;
    std::vector<float>::const_iterator target_it;
    uint64_t size;
    bool last_batch;
};

class DataReader {
public:
    DataReader(std::string filename, std::string config, uint64_t batchsize, 
            uint64_t bit_prec, std::string delim = ",", bool generate_names=true);
    void GetData();
    Batch* GetBatch();
    BatchIterator* GetBatchIterator();
    uint64_t GetNumericFeaturesNumber();
    uint64_t GetCatFeaturesNumber();
    std::vector<std::string> GetFeatureNames();
    void InitFeatureNames(std::vector<std::string> &names);

private:
    std::string file_name;
    std::string config_path;
    std::string delimeter;

    std::ifstream input_file;
    uint64_t buffer_size;
    uint64_t available_size;
    bool end_of_data_flag;
    bool end_of_file_flag;
    
    std::vector<std::vector<float>> data;
    std::vector<std::vector<uint64_t>> data_cat;
    std::vector<float> target;

    std::vector<size_t> cat_indices;
    std::vector<size_t> numeric_indices;
    std::vector<std::string> cat_feature_names;

    uint64_t current_index;
    uint64_t batch_size;
    uint64_t n_features;
    uint64_t bit_precision;

    uint64_t Hash(std::string s, uint64_t seed);
    uint64_t Murmur3(std::string s, uint64_t seed);
    void ReadConfig();
    void ComputeNumericIndices();
    void GenerateCatFeatureNames();
    std::string GenerateRandomString(size_t length);


    std::string chrs;
    std::mt19937 generator;
    std::uniform_int_distribution<size_t> pick;
    std::uniform_int_distribution<uint64_t> length_distribution;
};

void PrintBatch(Batch* batch);

#endif // DATA_READER_H
