import numpy as np
import pandas as pd
import argparse

def extract_and_save(args):
    csv = pd.read_csv(f'{args.path}/mnist_{args.mode}.csv')
    np_csv = csv.values
    print(f'Ready to extract {np_csv.shape} sized csv')
    print(f'It will be svaed to {args.save}')
    N = np_csv.shape[0]
    label = np_csv[:, 0]
    ims = np_csv[:, 1:].reshape(N, 28, 28)
    np.save(f'{args.save}/{args.mode}_mnist.npy', ims)
    np.save(f'{args.save}/{args.mode}_label.npy', label)
    print('Extraction DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-p",
            "--path",
            help="location of MNIST .csv file",
            type=str,
            default='./')
    parser.add_argument(
            "-s",
            "--save",
            help="where to save the MNIST .npy format",
            type=str,
            default='./')
    parser.add_argument(
            "-m",
            "--mode",
            help="choose which MNIST dataset [train | test]",
            default='train',
            choices=['train', 'test'])
    opt = parser.parse_args()

    extract_and_save(opt)
