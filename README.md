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
./yasli fit -i <input path> [optional parameters]
```
| Parameter   | Short option | Description | Default value  |
| ----------- |---------------| --------|  -----------------|
| --input-path|   -i  | The path to the input file with the dataset. |Required parameter  |
| --model-path|   -m  | The path to the file where to save the resulting model. | "model.bin" |
| --config|   -c  | The path to the config file with indices of categorical features. | "" (no categorical features) |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |
| --loss-function|   -l  |The metric to use in training. Possible values: "mse", "logistic". | "mse"  |
| --optimizer|   -O  | Optimizer.  Possible values: "sgd", "adagrad" | "sgd"  |
| --learning-rate|   -w  | The learning rate. | 0.1  |
| --epochs|   -e  | The number of times the algorithm will cycle over the data. |100 |
| --batch|   -B  | Batch size. | 64 |
| --bits|   -b  | The number of bits in the feature table. | 24 |


### Applying a model
```
./yasli apply  -i <input path>  -o <output path> [optional parameters]
```
| Parameter   | short option | Description | Default value  |
| ----------- |--------------| ------------| -------------- |
| --input-path|   -i  | The path to the input file with the dataset. |Required parameter  |
| --output-path|   -o  | The path to the output file with the predictions. |Required parameter  |
| --model-path|   -m  | The path to the file from where to load the model. | "model.bin" |
| --config|   -c  | The path to the config file with indices of categorical features. | "" (no categorical features) |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |
| --batch|   -b  | Batch size. | 64 |


### Benchmarking

#### Avazu Dataset (https://www.kaggle.com/c/avazu-ctr-prediction)

##### SGD
|               | Yasli        |Vowpal Wabbit|Liblinear       |
| ------------- |--------------| ------------| -------------- |
| fit time      |  4m57.680s    |   3m54.729s  |   12m2.031s    |
| apply time    |  3m4.265s  |    1m57.846s  |      -         |
| ROC AUC score |              |    0.65     |    0.6511019   |

##### Adaptive
|               | Yasli        |Vowpal Wabbit|Liblinear       |
| ------------- |--------------| ------------| -------------- |
| fit time      |  4m48.680s    |  3m41.199s  |   13m35.433s   |
| apply time    |  3m6.199s    |  2m0.575s   |      -         |
| ROC AUC score |   0.73076    |   0.735663  |   0.723977     |

#### Criteo Dataset (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)

##### SGD
|               | Yasli        |Vowpal Wabbit|Liblinear       |
| ------------- |--------------| ------------| -------------- |
| fit time      |  9m51.092s    |  4m4.163s  |   -    |
| apply time    |   2m59.514s |    2m23.315s  |      -         |
| ROC AUC score |    0.57402    |   0.56402     |    -   |

##### Adaptive
|               | Yasli        |Vowpal Wabbit|Liblinear     |
| ------------- |--------------| ------------| ------------ |
| fit time      |  9m48.680s   |  3m48.278s |   79m58.181s   |
| apply time    |  2m6.199s    |  2m1.829s |      -      |
| ROC AUC score |   0.73076    |   0.727919  |   0.4951   |

## Authors

* **Alexander Sakhnov**
* **Yury Kolomeytsev**
