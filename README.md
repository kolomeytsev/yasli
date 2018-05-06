# yasli
Yet Another Scalable Linear Model (project for LSML course at YSDA)

## Prerequisites

```
Boost
```

## Installing

```
g++ -std=c++11 yasli.cpp -o yasli
```

## Running

### Training a model

```
./yasli fit -i <input path> -m <model path>  [optional parameters]
```
| Parameter   | Short option | Description | Default value  |
| ----------- |---------------| --------|  -----------------|
| --input-path|   -i  | The path to the input file with the dataset. |Required parameter  |
| --model-path|   -m  | The path to the file where to save the resulting model. | "model.bin" |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |
| --loss-function|   -l  |The metric to use in training. Possible values: "mse", "logistic". | "mse"  |
| --optimizer|   -O  | Optimizer.  Possible values: "sgd", "adagrad" | "sgd"  |
| --learning-rate|   -w  | The learning rate. | 0.000001  |
| --epochs|   -e  | The number of times the algorithm will cycle over the data. |100 |
| --batch|   -b  | Size of a batch. | 64 |


### Applying a model
```
./yasli apply  -i <input path>  -m <model path> [optional parameters]
```
| Parameter   | short option | Description | Default value  |
| ----------- |--------------| ------------| -------------- |
| --input-path|   -i  | The path to the input file with the dataset. |Required parameter  |
| --output-path|   -i  | The path to the output file with the predictions. |Required parameter  |
| --model-path|   -m  | The path to the file from where to load the model. | "model.bin" |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |

## Authors

* **Alexander Sakhnov**
* **Yury Kolomeytsev**
