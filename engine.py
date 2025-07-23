import numpy as np


class Module:
    def forward(self, *args, **kwargs):
        return NotImplemented

    def backward(self, *args, **kwargs):
        return NotImplemented

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        return NotImplemented

    def zero_grad(self):
        return NotImplemented
        # self.grad = {k: np.zeros_like(v) for k, v in self.grad.items()}

