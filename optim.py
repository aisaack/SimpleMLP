import numpy as np
import nn


class Optim:
    def __init__(self, model: nn.Module, lr: float, weight_decay: float) -> None:
        assert isinstance(model, nn.Module)
        self.model = model
        self.lr = lr
        self.lamb = weight_decay

    def step(self, dL: np.ndarray) -> None:
        return NotImplemented

    def zero_grad(self) -> None:
        for n, m in self.model.named_modules():
            m.zero_grad()

    def weight_decay(self, param: np.ndarray) -> np.ndarray:
        return param * self.lamb
    


class SGD(Optim):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float=0.0):
        super().__init__(model, lr, weight_decay)
        
    def step(self, dL: np.ndarray) -> None:
        self.model.backward(dL)
        for n, m in self.model.named_modules():
            if hasattr(m, 'w'):
                # weight decay:
                # grad_{t} = grad_{t} + lambda * theta_{t} 
                if self.lamb != 0.0:
                    m.grad['dw'] += self.weight_decay(m.w)

                # parameter update:
                # theta_{t + 1} = theta_{t} - lr * grad_{t}
                m.w -= self.lr * m.grad['dw']

                if m.bias is not None:
                    if self.lamb != 0.0:
                        m.grad['db'] *= self.weight_decay(m.b)
                    m.b -= self.lr * m.grad['db']


class Adam(Optim):
    def __init__(self, model: nn.Module, lr: float, weight_decay: float=0.0,
                 beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:
        super().__init__(model, lr, weight_decay)
        assert beta1 < 1 and beta1 >= 0, 'range of beta1: [0. 1)'
        assert beta2 < 1 and beta2 >= 0, 'range of beta2: [0. 1)'
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
        # Both 1st and 2nd moment should not be scalar
        self.m = self.zero_moment() # 1st moment
        self.v = self.zero_moment() # 2nd moment

        # Iteration step.
        self.t = 0

    def step(self, dL: np.ndarray) -> None:
        self.t += 1
        self.model.backward(dL)
        for i, (n, m) in enumerate(self.model.named_modules()):
            k = f'{n}_{i}_'
            if hasattr(m, 'w'):
                if self.lamb != 0.0:
                    m.grad['dw'] += self.weight_decay(m.w)

                gw = m.grad['dw']
                self.m[k +'w'] *= self.beta1
                self.m[k +'w'] += (1 - self.beta1) * gw
                self.v[k +'w'] *= self.beta2
                self.v[k +'w'] += (1 - self.beta2) * (gw * gw)
                wm_hat_t = self.m[k + 'w'] / (1 - self.beta1 ** self.t)
                wv_hat_t = self.v[k + 'w'] / (1 - self.beta1 ** self.t)
                m.w -= self.lr * (wm_hat_t / (np.sqrt(wv_hat_t) + self.eps))

            if hasattr(m, 'b'):
                if m.bias is not None:
                    if self.weight_decay != 0.0:
                        m.grad['db'] += self.weight_decay(m.b)
            
                    gb = m.grad['db']
                    self.m[k +'b'] *= self.beta1
                    self.m[k +'b'] += (1 - self.beta1) * gb
                    self.v[k +'b'] *= self.beta2
                    self.v[k +'b'] += (1 - self.beta2) * (gb * gb)
                    bm_hat_t = self.m[k + 'b'] / (1 - self.beta1 ** self.t)
                    bv_hat_t = self.v[k + 'b'] / (1 - self.beta1 ** self.t)
                    m.b -= self.lr * (bm_hat_t / (np.sqrt(bv_hat_t) + self.eps))

    def zero_moment(self) -> np.ndarray:
        moment = {}
        for i, (n, m) in enumerate(self.model.named_modules()):
            k = f'{n}_{i}_'
            if hasattr(m, 'w'):
                moment[k + 'w'] = np.zeros_like(m.w)

            if hasattr(m, 'b'):
                if m.bias is not None:
                    moment[k + 'b'] = np.zeros_like(m.b)

        return moment


