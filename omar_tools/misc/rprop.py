import numpy as np


class RProp:

    def __init__(self, gradient, loss):
        self.gradient = gradient
        self.loss = loss
        self.epoch = None
        self.x = None
        self.loss_value = None
        self.delta = None
        self.g_new = self.g_old = None
        self.p = self.s = None
        self.results = None

    def converged(self, stop_loss):
        self.loss_value = self.loss(self.x)
        if self.loss_value <= stop_loss:
            self.results = {'sol': self.x, 'epoch': self.epoch, 'loss': self.loss_value}
            return True
        else:
            return False

    def update(self, k_plus, k_minus):
        self.g_new = self.gradient(self.x).copy()     # gradient might return x itself
        self.s = np.sign(self.g_new)
        self.p = np.sign(self.g_old * self.g_new)

        for j in range(len(self.x)):
            if self.p[j] < 0:
                self.delta[j] *= k_minus
                self.g_old[j] = 0
            else:
                if self.p[j] > 0:
                    self.delta[j] *= k_plus
                self.x[j] -= self.s[j] * self.delta[j]
                self.g_old[j] = self.g_new[j]

    def fit(self, x0: np.array, n_steps, stop_loss, k_plus=1.2, k_minus=.5, speed0=.1):
        self.epoch = 0
        self.x = x0.copy()
        self.delta = np.ones(len(x0)) * speed0
        self.g_old = np.zeros(len(x0))

        if self.converged(stop_loss):
            return self.results

        for self.epoch in range(n_steps):
            self.update(k_plus, k_minus)            # core of the algorithm

            if self.converged(stop_loss):
                return self.results

        else:
            raise ValueError('Algorithm failed to converge')


def test():
    def loss(v):
        return v.dot(v)

    def grad(v):
        return v

    alg = RProp(grad, loss)
    sol = alg.fit(x0=np.array([2., 4.]), n_steps=20, stop_loss=.1)
    print(f'sol = {sol["sol"]}')
    print(f'loss = {sol["loss"]:.2f}')
    print(f'epoch = {sol["epoch"]}')


if __name__ == '__main__':
    test()
