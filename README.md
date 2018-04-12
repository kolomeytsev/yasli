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
| ----------- |-------| --|  -------|
| --input-path|   -i  | The path to the input file with dataset. |Required parameter  |
| --model-path|   -m  | The path to the file where to save the resulting model. | Required parameter  |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |
| --loss-function|   -l  |The metric to use in training. Possible values: "mse", "adagrad". | "mse"  |
| --optimizer|   -O  | Optimizer.  Possible values: "sgd", "logistic". | "sgd"  |
| --learning-rate|   -w  | The learning rate. | 0.000001  |
| --iterations|   -I  | The maximum number of iterations of optimization. |5000000  |


### Applying a model
```
./yasli apply  -i <input path>  -m <model path> [optional parameters]
```
| Parameter   | short option | Description | Default value  |
| ----------- |-------| -----|  -----|
| --input-path|   -i  | The path to the input file with dataset. |Required parameter  |
| --model-path|   -m  | The path to the file from where to load the model. | Required parameter  |
| --delimiter|   -d  | The path to the input file with dataset. | ","  |

## Authors

* **Alexander Sakhnov**
* **Yury Kolomeytsev**
