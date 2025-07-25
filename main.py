import numpy as np

import nn
import optim
from model import Model
from dataset import DataLoader
from engine import Module
import argparse

def load_datasets(args):
    return {'train': DataLoader(train=True, batch_size=args.batch_size),
            'test': DataLoader(train=False, batch_size=args.batch_size)}

def accuracy(y, y_hat):
    p = nn.softmax_forward(y)
    high_p = np.argmax(p, axis=1)
    res = (high_p == y_hat)
    return res

def train_loop(args, model, dataset, loss_fn, optimizer):
    batch_loss = []
    for i, (x, y_hat) in enumerate(dataset):
        x = x.reshape(args.batch_size, args.input_feature)
        y = model(x)
        loss = loss_fn(y, y_hat)
        dL = loss_fn.backward()
        optimizer.step(dL)
        optimizer.zero_grad()
        batch_loss.append(loss.mean())
    return sum(batch_loss) / len(batch_loss)

def test_loop(args, model, dataset, loss_fn):
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
        mean_test_loss = sum(test_losses) / N
    return mean_test_loss, out_acc


def train(model, datasets, loss_fn, args):
    optimizer = getattr(optim, args.optim)(model, args.lr)

    for e in range(args.epochs):
        train_loss = train_loop(args, model, datasets['train'], loss_fn, optimizer) 
        print(f'# epochs: {e} ---  train loss: {train_loss:.3f}')
        if e % 10 == 0:
            test_loss, acc = test_loop(args, model, datasets['test'], loss_fn)
            print(f'test loss: {test_loss:.3f}')
            print(f'test acc: {acc:.3f}', )

    fin_loss, fin_acc = test_loop(args, model, datasets['test'], loss_fn)
    print(f'final loss: {fin_loss:.3f}')
    print(f'final acc: {fin_acc:.3f}', )

    
            

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs',
                       help='set how many epochs to run',
                       type=int,
                       default=10)
    parse.add_argument('--lr',
                       help='set learning rate',
                       type=float,
                       default=1e-3)
    parse.add_argument('--batch_size',
                       help='set batch size',
                       type=int,
                       default=32)
    parse.add_argument('--optim',
                       help='set optimimizer',
                       type=str,
                       default='SGD')
    parse.add_argument('--input_feature',
                       help='set input features',
                       type=int,
                       default=784)
    parse.add_argument('--num_class',
                       help='set number of class',
                       type=int,
                       default=10)
    parse.add_argument('--init_params',
                       help='set weight initialization method',
                       type=str,
                       default='xavier')

    
    opt = parse.parse_args()
    
    model = Model(
            inp_channel=opt.input_feature,
            num_class=opt.num_class,
            init_params=opt.init_params)
    datasets = load_datasets(opt)
    criterion = nn.CrossEntropy()
    train(model, datasets, criterion, opt)
