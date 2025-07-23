import numpy as np
from typing import Tuple
from engine import Module


def linear_forward(x: np.ndarray, w: np.ndarray, b: bool=None) -> Tuple:
    r'''
    Implement: y = xA.T + b
    compute above equations and cache x, w, and b
    '''
    cache = {'x': x, 'w':w}
    out = np.matmul(x, w.T)
    if b is not None:
        cache['b'] = b
        out += b
    return out, cache

def linear_backward(dout: np.ndarray, cache: dict) -> dict:
    x, w = cache['x'], cache['w']
    dx = np.matmul(dout, w)
    dw = np.matmul(dout.T, x)
    grad = {'dx': dx, 'dw': dw}
    if 'b' in cache.keys():
        db = np.sum(dout, axis=0)
        grad['db'] = db
    return grad


class Linear(Module):
    def __init__(self, in_channel: int, out_channel: int, bias: bool=None, dtype=np.float32) -> None:
        self.w = np.random.randn(out_channel, in_channel).astype(dtype)
        self.b = None
        self.grad = {'dw': np.zeros_like(self.w, dtype=dtype)}
        self.bias = bias
        if self.bias is True:
            self.b = np.ones((out_channel,), dtype=dtype)
            self.grad['db'] = np.zeros_like(self.b)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        out, cache = linear_forward(x, self.w, self.b)
        self.cache['w'] = cache['w']
        self.cache['x'] = cache['x']
        if self.bias is not None:
            self.cache['b'] = cache['b']
        return out
    
    def backward(self, dout) -> None:
        grad = linear_backward(dout, self.cache)
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
        

def log_softmax_forward(x):
    M = np.max(x, axis=-1, keepdims=True)
    log_sum_exp = M + np.log(np.sum(np.exp(x - M), axis=-1, keepdims=True))
    return x - log_sum_exp

def softmax_forward(x):
    M = np.max(x, axis=-1, keepdims=True)
    num = np.exp(x - M)
    den = np.sum(num, axis=-1, keepdims=True)
    return num / den

def softmax_backward(x):
    return

class Softamx(Module):
    def forward(x):
        return softmax_forward(x)

    def backward(din):
        return softmax_backward(din)


def crossentropy_forward(y, y_hat):
    # Applied Log-Sum-Exp trick.
    # It guarantees numerical stability.
    # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    n = y.shape[0]
    log_p = log_softmax_forward(y)
    # M = np.max(y, axis=-1, keepdims=True)
    # log_p = y - M - np.log(np.sum(np.exp(y - M), axis=-1, keepdims=True))
    loss_per_sample = -log_p[range(n), y_hat]
    batch_loss = np.sum(loss_per_sample) / n
    cache = {'y': y, 'y_hat': y_hat, 'p': log_p}
    return batch_loss, cache

def crossentropy_backward(dout, cache):
    y, y_hat, log_p = cache.values()
    N, C= y.shape
    
    # dy = y.copy()
    dy = np.exp(log_p)
    dy[np.arange(N), y_hat] -= 1
    dy = dy / N
    return dy * dout

class CrossEntropy(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, y, y_hat):
        out, cache = crossentropy_forward(y, y_hat)
        for k, v in cache.items():
            self.cache[k] = v
        return out

    def backward(self, dout=1.0):
        return crossentropy_backward(dout, self.cache)

    def parameters(self):
        return 

    def zero_grad(self):
        return


def relu_forward(x):
    cache = {'x': x}
    return np.maximum(0, x), cache

def relu_backward(dout, cache):
    x = cache['x']
    drelu = np.where(x > 0, 1, 0)
    return dout * drelu


class ReLU(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        out, cache = relu_forward(x)
        self.cache['x'] = cache['x']
        return out

    def backward(self, dout):
        grad = relu_backward(dout, self.cache)
        return grad

    def parameters(self):
        return 

    def zero_grad(self):
        return

def sigmoid_forward(x):
    out = 1 / (1 + np.exp(-x))
    cache = {'out': out}
    return out, cache

def stable_sigmoid_forward(x):
    pos_mask = (x > 0)
    out = np.zeros_like(x)
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    out[~pos_mask] = exp_x / (1 + exp_x)
    cache = {'out': out}
    return out, cache
 
def sigmoid_backward(dout, cache):
    sig = cache['out']
    dsig = sig * (1 - sig)
    return dout * dsig

class Sigmoid(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        out, cache = stable_sigmoid_forward(x)
        self.cache['out'] = cache['out']
        return out

    def backward(self, dout):
        grad = sigmoid_backward(dout, self.cache)
        return grad

    def parameters(self):
        return 

    def zero_grad(self):
        return


def layer_norm_forward(x, gamma, beta, esp=1e-7):
    mu = np.mean(x, axis=-1, keepdims=True)
    x_mu= x - mu
    var = np.var(x, axis=-1, keepdims=True)
    inv_var = np.pow(np.sqrt(var + esp), -1)
    x_hat = x_center * den
    out = gamma * x_hat + beta
    cache = {'x': x,
             'mu': mu,
             'var': var,
             'xmu': x_mu,
             'inv_var': inv_var,
             'x_hat': x_hat,
             'gamma': gamma,
             'beta': beta,
             'esp': esp}
    return out, cache

def layer_norm_backward(dout, cache):
    x, mu, var, x_mu, inv_var, x_hat, gamma, beta, esp = cache.values()
    dbeta = dout 
    dgamma_xhat= dout * x_hat
    dxhat = dout * gamma
    dgamma = dgamma_xhat * x_hat
    dxmu1 = dxhat * inv_var
    dinv_var = dxhat * x_mu

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
        out, cache = layer_norm_forward(x, self.gamma, self.beta, self.esp)
        self.cache = cache
        return out 
    
    def backward(self, dout):
        return layer_norm_backward(dout, self.cache, self.esp)


