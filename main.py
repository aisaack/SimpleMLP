import argparse
from typing import Tuple
import numpy as np

import nn
import functional as F
import optim
from model import Model
from dataset import DataLoader

def load_datasets(args) -> dict:
    return {'train': DataLoader(train=True, batch_size=args.batch_size, label_dim=0),
            'test': DataLoader(train=False, batch_size=args.batch_size, label_dim=0)}

def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    p = F.softmax_forward(y)
    if y.ndim == y_hat.ndim:
        y_hat = np.argmax(y_hat, axis=1)
    high_p = np.argmax(p, axis=1)
    res = (high_p == y_hat)
    return res

def train_loop(args, model: nn.Module, dataset: DataLoader,
               loss_fn: nn.CrossEntropy, optimizer: optim, e: int) -> float:
    batch_loss = []
    for i, (x, y_hat) in enumerate(dataset):
        x = x.reshape(args.batch_size, args.input_feature)
        y = model(x)
        loss = loss_fn(y, y_hat)
        dL = loss_fn.backward()
        if np.isnan(dL).any():
            print(f'# ----- {e} epoch, {i} iteration')
            print(f'derivative of loss: {dL}')
            print(f'from loss: {loss}')
            return StopIteration
        optimizer.step(dL)
        optimizer.zero_grad()
        batch_loss.append(loss.mean())
    return sum(batch_loss) / len(batch_loss)

def test_loop(args, model: nn.Module, dataset: DataLoader, loss_fn: nn.CrossEntropy) -> Tuple:
    N = len(dataset) * args.batch_size
    test_losses = []
    test_accs = []
    for i, (x, y_hat) in enumerate(dataset):
        y = model(x.reshape(args.batch_size, args.input_feature))
        loss = loss_fn(y, y_hat)
        acc = accuracy(y, y_hat)
        test_losses.append(loss)
        test_accs.append(acc)
        out_acc = np.sum(np.hstack(test_accs)) / N
        mean_test_loss = np.sum(test_losses) / N
    return mean_test_loss, out_acc


def train(model, datasets, loss_fn, args) -> None:
    optimizer = getattr(optim, args.optim)(model, args.lr, args.weight_decay)

    for e in range(args.epochs):
        train_loss = train_loop(args, model, datasets['train'], loss_fn, optimizer, e) 
        print(f'# epochs: {e} ---  train loss: {train_loss:.4f}')
        if e % 10 == 0:
            test_loss, acc = test_loop(args, model, datasets['test'], loss_fn)
            print(f'test loss: {test_loss:.4f}')
            print(f'test acc: {acc:.4f}', )

    fin_loss, fin_acc = test_loop(args, model, datasets['test'], loss_fn)
    print(f'final loss: {fin_loss:.4f}')
    print(f'final acc: {fin_acc:.4f}', )

    
            

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, default=10,
                       help='set how many epochs to run')

    parse.add_argument('--lr', type=float, default=1e-3,
                       help='set learning rate')

    parse.add_argument('--batch_size', type=int, default=32,
                       help='set batch size')

    parse.add_argument('--optim',type=str, default='Adam', choices=['SGD', 'Adam'],
                       help='set optimimizer [SGD | Adam]')

    parse.add_argument('--input_feature', type=int, default=784,
                       help='set input features')

    parse.add_argument('--num_class', type=int, default=10,
                       help='set number of class')

    parse.add_argument('--init_params', type=str, default='xavier',
                       help='set weight initialization method')
    
    parse.add_argument('--weight_decay', type=float, default=0.002,
                       help='set weight deca value')

    opt = parse.parse_args()
    
    model = Model(
            inp_channel=opt.input_feature,
            num_class=opt.num_class,
            init_params=opt.init_params)
    datasets = load_datasets(opt)
    criterion = nn.CrossEntropy()
    train(model, datasets, criterion, opt)
