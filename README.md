# Simple MLP
This is implementation of simple multi-layer perceptron (MLP) built with pure NumPy. The current focus is demonstration of core functionalities and numerical stability. It is shown decreasing loss over the 1st epoch.

## What is reimplemented:
**Custom layers**
* Linear (Dense)
* ReLU, and Sigmoid Activation
* Cross-Entropy loss

**Data handler**
* Custom `DataLoader` for data batch

**Optimizer**
* Stochastic gradient descent (SGD)

## Dataset preparation
Download MNIST dataset from the [link](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and extract `.npy` from `.csv` file.
```bash
python3 extract.py -s path/to/save/npy/files -d path/to/csv/files -m which/to/extract/train/or/test
```

## TODO
- [ ] Training / test loop 
- [ ] Add evaluation metric
