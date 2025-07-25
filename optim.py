import numpy as np
import nn

class SGD:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self, dL):
        self.model.backward(dL)
        for m in self.model.model:
            if hasattr(m, 'w'):
                m.w -= self.lr * m.grad['dw']
                if m.bias is not None:
                    m.b -= self.lr * m.grad['db']

    def zero_grad(self):
        for m in self.model.model:
            m.zero_grad()


