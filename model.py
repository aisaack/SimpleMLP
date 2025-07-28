import nn


class Model(nn.Module):
    def __init__(self, inp_channel=784, num_class=10, init_params='xavier'):
        self.model = [
                nn.Linear(inp_channel, 32, bias=True),
                nn.ReLU(),
                nn.Linear(32, 68, bias=True),
                nn.ReLU(),
                nn.Linear(68, 32, bias=True),
                nn.ReLU(),
                nn.Linear(32, num_class)]
        self.init_method = init_params
        self.init_params()

    def forward(self, x):
        y = x
        for m in self.model:
            y = m(y)
        return y

    def backward(self, dL):
        dout = dL
        for m in self.model[::-1]:
            dout = m.backward(dout)

    def named_modules(self):
        for m in self.model:
            n = m.__class__.__name__
            yield (n, m)

    def init_params(self):
        for _, m in self.named_modules():
            if hasattr(m, 'w'):
                getattr(m, 'init_' + self.init_method)()


if __name__ == '__main__':
    model = Model()
    for n, m in model.named_modules():
        print(n)
    
