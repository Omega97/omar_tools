"""
 "Integration" is a class that is useful for performing fast and precise integration in many dimensions.
 By initializing

alg = Integration(order, n_dimensions, repeat)

 we may perform integration repeatedly with different functions and od different domains

y1 = alg(f, v0, v1)
y2 = alg(g, v0, v1)
"""
import numpy as np
from itertools import product


def _matrix_for_finding_coefficients(size):
    m = np.zeros((size, size))
    for i in range(1, size):
        for j in range(size):
            m[i, j] = i ** j
    m[0, 0] = 1
    return m


def _array_for_integration(size):
    return np.array([(size-1) ** (i + 1) / (i + 1) for i in range(size)])


class BlockIntegral1D:
    """
    Integration algorithm of a given order in 1-D
    The same set of weights is used only one once (therefore on a "block")
    """
    def __init__(self, order):
        assert order >= 1
        self.order = order
        self.n = 1 + self.order
        self.w = None

        self.compute_weights()

    def compute_weights(self):
        m = _matrix_for_finding_coefficients(size=self.n)
        m_inv = np.linalg.inv(m)
        v = _array_for_integration(size=self.n)
        self.w = v.dot(m_inv)


class BlockIntegralND(BlockIntegral1D):
    """
    Integration algorithm of a given order in N-D
    The same set of weights is used only one once (therefore on a "block")
    """
    def __init__(self, order, n_dimensions):
        super().__init__(order)
        self.n_dimensions = n_dimensions
        self.w_nd = None
        self.compute_weights_nd()

    def compute_weights_nd(self):
        self.w_nd = np.zeros(shape=[self.order+1] * self.n_dimensions)
        for j in product(*[range(self.order+1) for _ in range(self.n_dimensions)]):
            self.w_nd.itemset(j, np.product([self.w[i] for i in j]))        # todo write better?


class Integration(BlockIntegralND):
    """
    Integration algorithm of a given order in N-D
    The same set of weights is used repeatedly on the whole domain
    Call this object with function fun and extremes x0, x1 to evaluate integral on a N-interval
    """
    def __init__(self, order, n_dimensions, repeat):
        super().__init__(order, n_dimensions)
        self.repeat = repeat
        self.w_int = None
        self.compute_integration_weights()

    def compute_integration_weights(self):
        length = self.order * self.repeat + 1
        self.w_int = np.zeros(shape=[length] * self.n_dimensions)

        for i in product(*[range(self.repeat) for _ in range(self.n_dimensions)]):
            for j in product(*[range(self.order+1) for _ in range(self.n_dimensions)]):
                i_ = np.array(i)
                j_ = np.array(j)
                index_ = i_ * self.order + j_
                self.w_int.itemset(tuple(index_), self.w_int.item(*index_) + self.w_nd.item(*j_))

    def __call__(self, fun, v0: np.array, v1: np.array):
        if not len(v0) == len(v1) == self.n_dimensions:
            raise IndexError(f'v0 and v1 must have length {self.n_dimensions} ({len(v0)} and {len(v0)} found instead)')

        out = 0.
        dv = np.product(v1 - v0) / self.order ** self.n_dimensions

        for indices in product(*(range(self.order * self.repeat + 1) for _ in range(self.n_dimensions))):
            x = v0 + (v1-v0) * np.array(indices) / (self.order * self.repeat)
            out += fun(*x) * self.w_int[indices]

        return out * dv / self.repeat ** self.n_dimensions
