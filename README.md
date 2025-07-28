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
```bash
# epochs: 0 ---  train loss: 1.923
test loss: 0.037
test acc: 0.524
# epochs: 1 ---  train loss: 1.120
# epochs: 2 ---  train loss: 0.988
# epochs: 3 ---  train loss: 0.880
# epochs: 4 ---  train loss: 0.766
# epochs: 5 ---  train loss: 0.697
# epochs: 6 ---  train loss: 0.634
# epochs: 7 ---  train loss: 0.598
# epochs: 8 ---  train loss: 0.556
# epochs: 9 ---  train loss: 0.522
final loss: 0.017
final acc: 0.810
```
