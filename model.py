import nn
from engine import Module

class Model(Module):
    def __init__(self, inp_channel=784, num_class=10):
        self.model = [
                nn.Linear(inp_channel, 32, bias=True),
                nn.ReLU(),
                nn.Linear(32, 68, bias=True),
                nn.ReLU(),
                nn.Linear(68, 32, bias=True),
                nn.ReLU(),
                nn.Linear(32, num_class)]

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

    def init_xavier(self):
        params = self.parameters()

