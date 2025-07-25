import numpy as np


class DataLoader:
    def __init__(self, data_dir='../MNIST_CSV', batch_size=32, 
                train=True, shuffle=True, drop_last=True):
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
        self.label  = np.load(label_path).astype(np.int32)
        # self.label = self.build_label(self.label)
        
        self.N = self.label.shape[0]
        self.num_batches = self.N // batch_size
    
    def __iter__(self):
        if self.shuffle is True:
            iidx = np.arange(self.N)
            np.random.shuffle(iidx)  # inplace processing
            self.__curr_im = self.im[iidx]
            self.__curr_label = self.label[iidx]
        else:
            self.__curr_im = self.im
            self.__curr_label = self.label

        if self.drop_last is True:
            N = self.num_batches * self.batch_size
            self.__curr_im = self.im[:N]
            self.__curr_label = self.label[:N]

        for i in range(len(self)):
            yield(self[i])

    def __getitem__(self, idx):
        s = idx * self.batch_size
        e = s + self.batch_size
        im = self.__curr_im[s:e]
        label = self.__curr_label[s:e]
        return im, label

    def __len__(self):
        return self.num_batches

    def build_label(self, y_hat):
        N = np.arange(y_hat.shape[0])
        out_y_hat = np.zeros((len(N), 10))
        out_y_hat[N, y_hat] = 1
        return out_y_hat


if __name__ == '__main__':
    dataset = DataLoader(train=True)
    print(len(dataset))
