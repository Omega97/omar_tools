""" lib of useful functions """
from math import exp, factorial
import numpy as np


def line(m, q):
    """return line function of given angular coefficient and y-intercept"""
    def f(x):
        return m * x + q
    return f


def gauss(average=0., std=1.):
    """normal distribution"""
    def f(x):
        return np.exp(-(x-average)**2/(2*std**2)) / ((2 * np.pi) ** .5 * std)
    return f


def norm(v, p=2):
    return sum([i**p for i in v]) ** (1/p)


def dist(v, w=None):
    """distance between two vectors"""
    if w is None:
        return norm(v)
    else:
        return sum([(v[i] - w[i])**2 for i in range(len(v))])**.5


def poly(x, coefficients):
    """ polynomial """
    return sum([x ** n * coefficients[n] for n in range(len(coefficients))])


def sigmoid(x=None):
    return 1/(1 + exp(-x))


def relu(x=None):
    return max(x, 0)


def prelu(x=None, p=0.):
    assert 0. <= p <= 1.
    return max(x, p * x)


def add_functions(v):
    """ return function, sum of all functions in v """
    def f(x_):
        out = 0
        for i in v:
            out += i(x_)
        return out
    return f


def product_function(v):
    """ return function, product of all functions in v """
    def f(x_):
        out = 1
        for i in v:
            out *= i(x_)
        return out
    return f


def softmax(v):
    """softmax"""
    out = [exp(i) for i in v]
    s = sum(out)
    return [i/s for i in out]


def double_factorial(n):
    """n!!"""
    assert type(n) is int
    if n < 0:
        raise ValueError('n must not be negative')
    elif n == 0:
        return 1
    elif n % 2:
        return np.product([i for i in range(1, n+1) if i % 2 == 1])
    else:
        return np.product([i for i in range(2, n+2) if i % 2 == 0])


def gamma_of_half_n(n):
    """gamma(n/2) for integer n > 0"""
    assert type(n) is int
    assert n > 0

    if n == 1:
        return np.pi ** .5
    elif n % 2:
        return double_factorial(n - 2) / 2 ** ((n - 1) / 2) * np.pi ** .5
    else:
        return factorial(n // 2 - 1)
