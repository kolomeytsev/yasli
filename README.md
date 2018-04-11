# yasli
Yet Another Scalable Linear Model (project for LSML course at YSDA)

### Prerequisites

```
Boost
```

### Installing

```
g++ -std=c++11 yasli.cpp -o yasli
```

## Running 

```
./yasli fit -i test_data.csv -l mse -O sgd -w 0.000001 -I 5000000

./yasli fit -i  test_data.csv -l mse -O adagrad -w 1 -I 5000000
```


## Authors

* **Alexander Sakhnov**
* **Yury Kolomeytsev**
