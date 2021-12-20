""" lib of useful functions """
from math import exp, tanh
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


def identity(x=None, der=False):
    if not der:
        return x
    else:
        def f(_):
            return 1
        return f


def sigmoid(x=None, der=False):
    if not der:
        return 1/(1 + exp(-x))
    else:
        def f(x_):
            return sigmoid(x_) * (1 - sigmoid(x_))
        return f


def th(x=None, der=False):
    if not der:
        return tanh(x)
    else:
        def f(x_):
            return 1 - th(x_) ** 2
        return f


def step(x=None, der=False):
    if not der:
        return 1 if x >= 0 else 0
    else:
        def f(_):
            return 0
        return f


def relu(x=None, der=False):
    if not der:
        return max(x, 0)
    else:
        def f(x_):
            return step(x_)
        return f


def prelu(x=None, p=0., der=False):
    assert 0. <= p <= 1.
    if not der:
        return max(x, p * x)
    else:
        def f(_):
            return max(1., p)
        return f


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
