import nn
from engine import Module

class Model(Module):
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

    def parameters(self):
        params = {}
        for i, m in enumerate(self.model):
            if isinstance(m, nn.Linear):
                n, p = m.parameters()
                params[n + f'_{i}'] = p
        return params

    def init_params(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                getattr(m, 'init_' + self.init_method)()

