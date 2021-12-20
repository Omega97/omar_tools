import numpy as np
from itertools import product


def that_matrix(size):
    m = np.zeros((size, size))
    for i in range(1, size):
        for j in range(size):
            m[i, j] = i ** j
    m[0, 0] = 1
    return m


def that_array(size):
    return np.array([(size-1) ** (i + 1) / (i + 1) for i in range(size)])


def count_interfaces(indices, order, repeat):
    return sum(False if i in (0, order * repeat) else i % order == 0 for i in indices)


def gamma_of_half_integer(x):
    if x > 1:
        return gamma_of_half_integer(x-1) * (x-1)
    elif x == 1.:
        return 1
    elif x == .5:
        return np.pi ** .5
    else:
        raise ValueError(f'gamma_of_half_integer({x}) is not defined')


def n_ball_surface_area(n):
    return np.pi ** (n/2) / gamma_of_half_integer(n/2 + 1) * n


class BlockIntegral1D:
    """
    1-D integration of a given order
    call this object with function fun and extremes x0, x1 to evaluate integral
    """

    def __init__(self, order):
        assert order >= 1
        self.order = order
        self.n = 1 + self.order
        self.w = None
        self.compute_weights()

    def compute_weights(self):
        m = that_matrix(size=self.n)
        m_inv = np.linalg.inv(m)
        v = that_array(size=self.n)
        self.w = v.dot(m_inv)

    def __call__(self, fun, x0: float, x1: float):
        dx = (x1 - x0) / self.order
        return np.sum([fun(x0 + dx * i) * self.w[i] for i in range(self.n)]) * dx


class BlockIntegralND(BlockIntegral1D):

    def __call__(self, fun, v0: np.array, v1: np.array):
        assert len(v0) == len(v1)
        dim = len(v0)
        out = 0.
        dv = np.product(v1 - v0) / self.order ** dim

        for indices in product(*(range(self.n) for _ in range(dim))):
            x = v0 + (v1-v0) * np.array(indices) / self.order
            w = np.product([self.w[indices[i]] for i in range(dim)])
            out += fun(x) * w
        return out * dv


class Integration(BlockIntegralND):

    def __init__(self, order, repeat):
        super().__init__(order)
        self.repeat = repeat

    def get_weight(self, indices, dim):
        out = np.product([self.w[indices[i] % self.order] for i in range(dim)])
        return out * 2 ** count_interfaces(indices, self.order, self.repeat)

    def __call__(self, fun, v0: np.array, v1: np.array):
        assert len(v0) == len(v1)
        dim = len(v0)
        out = 0.
        dv = np.product(v1 - v0) / self.order ** dim

        for weight_indices in product(*(range(self.order * self.repeat + 1) for _ in range(dim))):
            i_ = np.array(weight_indices)
            x = v0 + (v1-v0) * i_ / (self.order * self.repeat)
            out += fun(x) * self.get_weight(weight_indices, dim)

        return out * dv / self.repeat ** dim


class SphericalIntegration:

    def __init__(self, order, repeat):
        self.algorithm = Integration(order, repeat)

    def __call__(self, fun, r0, epsilon_0, epsilon_inf, max_itr=100):
        out = 0.

        def new_f(x):
            n = len(x)
            return fun(x) * (x.dot(x)) ** ((n-1)/2) * n_ball_surface_area(n)

        for i in range(max_itr):
            delta_out = self.algorithm(new_f, v0=..., v1=...)
            out += delta_out

            if delta_out <= epsilon_inf:
                return out
        else:
            raise ValueError('Integral does not converge!')


def test_1():
    alg = BlockIntegral1D(order=5)
    print(alg.w)

    out = alg(np.sin, 0, np.pi/2)
    print(out)


def test_2():
    v0 = np.array([1, 3])
    v1 = np.array([2, 4])
    alg = BlockIntegralND(order=3)

    def f(x):
        r_2 = x.dot(x)
        return r_2 * 3

    out = alg(f, v0, v1)
    print(out)


def test_3():
    v0 = np.array([0, 0])
    v1 = np.array([4, 4])
    alg = Integration(order=5, repeat=5)

    def f(x):
        r_2 = x.dot(x)
        # return r_2 * 3
        return np.exp(-r_2) / np.pi * 4

    out = alg(f, v0, v1)
    print(out)


if __name__ == '__main__':
    test_3()
