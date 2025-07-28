import numpy as np
from typing import Tuple

import functional as F


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

    def update_cache(self, cache):
        for k, v in cache.items():
            self.cache[k] = v


class Linear(Module):
    def __init__(
            self, in_channel: int, out_channel: int,
            bias: bool=None, dtype=np.float32, init_weight='he') -> None:
        self.w = np.random.randn(out_channel, in_channel).astype(dtype)
        self.b = None
        self.grad = {'dw': np.zeros_like(self.w, dtype=dtype)}
        self.bias = bias
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dtype = dtype
        if self.bias is True:
            self.b = np.ones((out_channel,), dtype=dtype)
            self.grad['db'] = np.zeros_like(self.b)
        self.cache = {}

        getattr(self, 'init_' + init_weight)()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out, cache = F.linear_forward(x, self.w, self.b)
        self.update_cache(cache)
        return out
    
    def backward(self, dout) -> None:
        grad = F.linear_backward(dout, self.cache)
        self.grad['dw'] += grad['dw']
        if self.bias is not None:
            self.grad['db'] += grad['db']
        return grad['dx']

    def parameters(self):
        name = self.__class__.__name__
        params = {'w': self.w}
        if self.b is not None:
            params['b'] = self.b
        return name, params

    def zero_grad(self):
        self.grad['dw'].fill(0)
        if self.bias is not None:
            self.grad['db'].fill(0)
    
    def init_xavier(self):
        w = np.random.randn(self.out_channel, self.in_channel).astype(self.dtype)
        self.w = w * np.sqrt(2 / (self.in_channel + self.out_channel))

    def init_he(self):
        w = np.random.randn(self.out_channel, self.in_channel).astype(self.dtype)
        self.w = w * np.sqrt(2 / self.in_channel)

    def init_alex(self):
        w = np.random.randn(self.out_channel, self.in_channel).astype(self.dtype)
        self.w = w * 0.01

        
class Softamx(Module):
    def forward(x):
        return F.softmax_forward(x)

    def backward(din):
        return F.softmax_backward(din)

class CrossEntropy(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, y, y_hat):
        out, cache = F.naive_crossentropy_forward(y, y_hat)
        self.update_cache(cache)
        return out

    def backward(self, dout=1.0):
        return F.naive_crossentropy_backward(dout, self.cache)

    def parameters(self):
        return 

    def zero_grad(self):
        return


class ReLU(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        out, cache = F.relu_forward(x)
        self.cache['x'] = cache['x']
        return out

    def backward(self, dout):
        grad = F.relu_backward(dout, self.cache)
        return grad

    def parameters(self):
        return 

    def zero_grad(self):
        return


class Sigmoid(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        out, cache = F.stable_sigmoid_forward(x)
        self.cache['out'] = cache['out']
        return out

    def backward(self, dout):
        grad = F.sigmoid_backward(dout, self.cache)
        return grad

    def parameters(self):
        return 

    def zero_grad(self):
        return


class LayerNorm(Module):
    def __init__(self, in_channel, esp=1e-7):
        self.cache = None
        self.gamma = np.ones()
        self.beta = np.zeros()
        self.esp = esp
        self.xmean = None
        self.xvar = None

    def forward(self, x):
        out, cache = F.layer_norm_forward(x, self.gamma, self.beta, self.esp)
        self.cache = cache
        return out 
    
    def backward(self, dout):
        return F.layer_norm_backward(dout, self.cache, self.esp)

