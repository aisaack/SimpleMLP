import numpy as np


class DataLoader:
    def __init__(self, data_dir='../MNIST_CSV', batch_size=32, train=True, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if train is True:
            data_path = f'{data_dir}/train_mnist.npy'
            label_path = f'{data_dir}/train_label.npy'
        else:
            data_path = f'{data_dir}/test_mnist.npy'
            label_path = f'{data_dir}/test_label.npy'

        self.im = np.load(data_path).astype(np.float32)
        self.label = np.load(label_path).astype(np.int32)
        self.N = self.label.shape[0]
        self.num_batches = self.N // batch_size

    def __getitem__(self, idx):
        if idx == 0:
            if self.shuffle is True:
                iidx = np.arange(self.N)
                np.random.shuffle(iidx)  # inplace processing
                self.im = self.im[iidx]
                self.label = self.label[iidx]

            if self.drop_last is True:
                N = (self.N // self.batch_size) * self.batch_size
                self.im = self.im[:N]
                self.label = self.label[:N]

        if idx >= self.num_batches:
            raise StopIteration

        s = idx * self.batch_size
        e = (idx + 1) * self.batch_size
        im = self.im[s:e]
        label = self.label[s:e]
        return im, label

    def __len__(self):
        return self.label.shape[0]


if __name__ == '__main__':
    dataset = DataLoader()
    print(len(dataset))
