# yasli
Yet Another Scalable Linear Model (project for LSML course at YSDA).

Ясли - система для обучения линейных моделей, которая была создана в качестве проекта по курсу "Машинное обучение на больших данных" в ШАДе Яндекса.

### Особенности:
 1. Out-of-core обучение.
 2. Поддержка задач:
   * Бинарная классификация
   * Регрессия
 2. Поддержка оптимизаторов:
   * SGD
   * Adagrad
   * FTRL-Proximal
 3. Поддержка функций потерь:
   * MSE (регрессия)
   * Logistic (классификация)
 4. Хэширование категориальных признаков.

Данная программа имеет два режима работы - обучение (fit) и применение (apply). Для обоих режимов формат входных данных одинаковый.

### Формат входных данных:
  - 1-й столбец - метки классов (-1 или 1) или числовое значение ответов регрессии.
  - Остальные столбцы - численные или строковые значения признаков через разделитель, который передается через флаг --delimiter (-d).
  - Категориальные признаки должны быть отмечены в конфигурационном файле, как индексы столбцов во входном файле, начиная с 0 (0-ой индекс имеет 1-й столбец с метками). Путь до конфиг файла передется через флаг --config (-c).

Ниже располагается информация об установке программы, о запуске обучения и применения со всеми флагами, а также результаты сравнения ее с другими аналогами: vowpal wabbit и liblinear на датасетах Criteo и Avazu.

## Prerequisites
```
Boost
```

## Compiling
To build the model on Unix systems just type:
```
$ make
```

## Running

### Training a model

```
$ ./yasli fit -i <input path> [optional parameters]
```
| Parameter   | Short option | Description | Default value  |
| ----------- |---------------| --------|  -----------------|
| --input-path|   -i  | The path to the input file with the dataset. |Required parameter  |
| --model-path|   -m  | The path to the file where to save the resulting model. | "model.bin" |
| --config|   -c  | The path to the config file with indices of categorical features. | "" (no categorical features) |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |
| --loss-function|   -l  |The metric to use in training. Possible values: "mse", "logistic". | "mse"  |
| --optimizer|   -O  | Optimizer.  Possible values: "sgd", "adagrad", "ftrl" | "sgd"  |
| --learning-rate|   -w  | The learning rate. | 0.1  |
| --epochs|   -e  | The number of times the algorithm will cycle over the data. |100 |
| --batch|   -B  | Batch size. | 64 |
| --bits|   -b  | The number of bits in the feature table. | 24 |
| --ftrl_alpha|  no short  | Ftrl alpha parameter.  | 0.005 |
| --ftrl_beta|   no short   | Ftrl beta parameter . | 0.1 |
| --l1|   no short  | L1 regularization parameter for ftrl method. | 0 |
| --l2|   no short  | L2 regularization parameter for ftrl method. | 0 |


### Applying a model
```
$ ./yasli apply  -i <input path>  -o <output path> [optional parameters]
```
| Parameter   | short option | Description | Default value  |
| ----------- |--------------| ------------| -------------- |
| --input-path|   -i  | The path to the input file with the dataset. |Required parameter  |
| --output-path|   -o  | The path to the output file with the predictions. |Required parameter  |
| --model-path|   -m  | The path to the file from where to load the model. | "model.bin" |
| --config|   -c  | The path to the config file with indices of categorical features. | "" (no categorical features) |
| --delimiter|   -d  | Delimiter that divides values in the input file. | ","  |
| --batch|   -b  | Batch size. | 64 |


## Benchmarking

В данном разделе располагается сравнение 3-х систем для обучения линейных моделей: Yasli, Vowpal Vabbit, Liblinear.

В файле **_prepare_data_and_benchmarking.ipynb_** расположен код, необходимый для подготовки датасетов в нужный формат для каждой системы, а также запуск и результаты сравнения.

### Avazu Dataset (https://www.kaggle.com/c/avazu-ctr-prediction)

##### SGD
|               | Yasli        |Vowpal Wabbit|Liblinear       |
| ------------- |--------------| ------------| -------------- |
| fit time      |  4m44.840s    |   4m15.223s  | 15m16.911s   |
| apply time    |  2m3.056s |    2m8.978s |     0m23.646s        |
| ROC AUC score |   0.72548     |   0.727919  |   0.53872  |

##### Adaptive
|               | Yasli        |Vowpal Wabbit|
| ------------- |--------------| ------------|
| fit time      | 4m34.449s   |  3m41.199s  |
| apply time    |  2m3.847s    |  2m0.575s   |
| ROC AUC score |   0.731246    |   0.735663 |

##### FTRL
|               | Yasli        |Vowpal Wabbit  |
| ------------- |--------------| ------------|
| fit time      |  3m39.782s    |  3m44.515s  |  
| apply time    |  2m3.847s    |  2m0.287s  |  
| ROC AUC score |   0.73076    |  0.7313928 | 

### Criteo Dataset (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)

##### SGD
|               | Yasli        |Vowpal Wabbit|Liblinear    |
| ------------- |--------------| ------------| -------------- |
| fit time      |  9m10.360s   |  4m4.163s  |   190m30.584s   |
| apply time    |   2m58.486s |   2m24.252s |      0m47.897s         |
| ROC AUC score |    0.61391   |   0.56402     |    0.63074   |

##### Adaptive
|               | Yasli        |Vowpal Wabbit|
| ------------- |--------------| ------------|
| fit time      |  9m46.675s   |  3m48.278s |
| apply time    |  2m59.447s   |  2m23.315s |
| ROC AUC score |   0.71296    |   0.77459475  |

##### FTRL
|               | Yasli        |Vowpal Wabbit  |
| ------------- |--------------| ------------|
| fit time      |  7m29.298s   |  3m59.173s |  
| apply time    |  2m50.992s    | 2m18.430s |  
| ROC AUC score |   0.767993    |  0.768009| 

## Authors

* **Alexander Sakhnov**
* **Yury Kolomeytsev**
