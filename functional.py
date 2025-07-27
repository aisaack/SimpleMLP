import numpy as np
from typing import Tuple


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

def log_softmax_forward(x: np.ndarray) -> np.ndarray:
    M = np.max(x, axis=-1, keepdims=True)
    log_sum_exp = M + np.log(np.sum(np.exp(x - M), axis=-1, keepdims=True))
    return x - log_sum_exp

def softmax_forward(x):
    stable_x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(stable_x)
    den = np.sum(ex, axis=-1, keepdims=True)
    return ex / den

def softmax_backward(x):
    return

def naive_crossentropy_forward(y, y_hat):
    eps = 1e-15
    p = softmax_forward(y)
    safe_y = np.clip(p, eps, 1-eps)
    cache = {'y': safe_y, 'y_hat': y_hat}
    if y.shape == y_hat.shape:
        loss = -np.sum(y_hat * np.log(safe_y), axis=-1)
        return loss.mean(), cache
    n = y.shape[0]
    log_p = -np.log(safe_y)
    loss = log_p[range(n), y_hat] / n
    return loss, cache

def naive_crossentropy_backward(dout, cache):
    y, y_hat, = cache.values()
    if y.shape == y_hat.shape:
        dy = (y - y_hat) / y.shape[0]
        return dy * dout
    n = y.shape[0]
    dy = y.copy()
    dy[np.arange(n), y_hat] -= 1
    dy /= n
    return dy * dout

def crossentropy_forward(y, y_hat):
    n = y.shape[0]
    p = softmax_forward(y)
    M = np.max(y, axis=-1, keepdims=True)
    log_p = y - M - np.log(np.sum(np.exp(y - M), axis=-1, keepdims=True))
    loss_per_sample = -log_p[range(n), y_hat]
    batch_loss = (loss_per_sample / n).mean()
    cache = {'y': y, 'y_hat': y_hat, 'p': p}
    return batch_loss, cache

def crossentropy_backward(dout,cache):
    y, y_hat, p = cache.values()
    N, C= y.shape
    
    dy = y.copy()
    dy[np.arange(N), y_hat] -= 1
    dy /= N
    return dy * dout

def stable_crossentropy_forward(y, y_hat):
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

def stable_crossentropy_backward(dout, cache):
    y, y_hat, log_p = cache.values()
    N, C= y.shape
    
    # dy = y.copy()
    dy = np.exp(log_p)
    dy[np.arange(N), y_hat] -= 1
    dy = dy / N
    return dy * dout

def relu_forward(x):
    cache = {'x': x}
    return np.maximum(0, x), cache

def relu_backward(dout, cache):
    x = cache['x']
    drelu = np.where(x > 0, 1, 0)
    return dout * drelu

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

