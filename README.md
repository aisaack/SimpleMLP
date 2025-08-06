# Simple MLP
This is implementation of simple multi-layer perceptron (MLP) built in pure NumPy. The current focus is demonstration of core functionalities and numerical stability.

## What is reimplemented:
**Custom layers**
* Linear (Dense)
* ReLU, and Sigmoid Activation
* Cross-Entropy loss

**Weight initialization**
* Alex
* Xavier
* He

**Data handler**
* Custom `DataLoader` for data batch

**Optimizer**
* Stochastic gradient descent (SGD)
* Adam

## Dataset preparation
Download MNIST dataset from the [link](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and extract `.npy` from `.csv` file.
```bash
python3 extract.py -s path/to/save/npy/files -p path/to/csv/files -m which/to/extract/train/or/test
```

## Run
```bash
python3 main.py --epochs 10 --batch_size 32 --lr 1e-3 --init_params xavier
```

## Result
```none
# epochs: 0 ---  train loss: 0.0116
test loss: 0.0096
test acc: 0.9066
# epochs: 1 ---  train loss: 0.0076
# epochs: 2 ---  train loss: 0.0067
# epochs: 3 ---  train loss: 0.0061
# epochs: 4 ---  train loss: 0.0058
# epochs: 5 ---  train loss: 0.0055
# epochs: 6 ---  train loss: 0.0053
# epochs: 7 ---  train loss: 0.0051
# epochs: 8 ---  train loss: 0.0049
# epochs: 9 ---  train loss: 0.0048
final loss: 0.0089
final acc: 0.9212
```
