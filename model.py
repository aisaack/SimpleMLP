import nn


class Model(nn.Module):
    def __init__(self, inp_channel=784, num_class=10, init_params='xavier'):
        self.model = [
                nn.Linear(inp_channel, 32, bias=True, init_weight=init_params),
                nn.ReLU(),
                nn.Linear(32, 68, bias=True, init_weight=init_params),
                nn.ReLU(),
                nn.Linear(68, 32, bias=True, init_weight=init_params),
                nn.ReLU(),
                nn.Linear(32, num_class, init_weight=init_params)]
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
                if self.init_method == m.init_weight:
                    inv = 1.0
                elif m.init_weight == 'alex':
                    inv = 100
                elif m.init_weight== 'xavier':
                    inv = np.sqrt((m.in_channel + n.out_channel) * (2**-1))
                elif m.init_weight== 'he':
                    inv = np.sqrt(n.in_channel * (2**-1))
                m.w *= inv
                getattr(m, 'init_' + self.init_method)()


if __name__ == '__main__':
    model = Model()
    for n, m in model.named_modules():
        print(n)
    
