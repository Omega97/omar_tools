"""
This module makes pyplot easier to use
"""
__author__ = "Omar Cusma Fait"
__date__ = (19, 12, 2021)
__version__ = "1.0.0"

import matplotlib.pyplot as plt
import numpy as np


def extended(array, extend_ratio=None, x_min=None, x_max=None, ):
    """
    extended domain for fit purposes, adds new value at head and tail of array
    :param array: array-link object
    :param extend_ratio: make domain longer in both directions
    :param x_min: new minimum
    :param x_max: new maximum
    :return: extended domain
    """
    assert array is not None
    out = list(array)
    x0 = None
    x1 = None
    if extend_ratio is not None:
        x0 = array[0] - (array[-1] - array[0]) * extend_ratio
        x1 = array[-1] + (array[-1] - array[0]) * extend_ratio
    if x_min is not None:
        if x_min < array[0]:
            x0 = x_min
    if x_max is not None:
        if x_max > array[-1]:
            x1 = x_max

    if x0 is not None:
        out = [x0] + out
    if x1 is not None:
        out = out + [x1]
    return out


class Plot:

    def __init__(self, ax=None):
        self.ax = ax

        self.x_plot = None
        self.y_plot = None
        self.x_scatter = None
        self.y_scatter = None
        self.x_hist = None
        self.y_hist = None

        self._plot = None
        self._scatter = None
        self._errorbar = None

    def sca(self):
        if self.ax is not None:
            plt.sca(self.ax)

    def scatter(self, x, y, label='data', zorder=2, **kwargs):
        self.sca()
        self.x_scatter = np.array(x)
        self.y_scatter = np.array(y)
        self._scatter = plt.scatter(x, y, label=label, zorder=zorder, **kwargs)

    def plot(self, x, y=None, func=None, linestyle='--', label='fit', **kwargs):
        """used to make fit"""
        self.sca()
        if y is None:
            assert func
            y = [func(i) for i in x]
        self.x_plot = np.array(x)
        self.y_plot = np.array(y)
        self._plot = plt.plot(x, y, linestyle=linestyle, label=label, **kwargs)

    def errorbar(self, x, y, yerr=None, xerr=None, fmt='none',
                 solid_capstyle='projecting', capsize=3, label='errors', **kwargs):
        self.sca()
        self._errorbar = plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, solid_capstyle=solid_capstyle,
                                      capsize=capsize, label=label, **kwargs)

    def hist(self, data, x_range, delta_x, extend=False, linewidth=1.2, density=True, edgecolor='black'):
        """
        - plot histogram
        - set self.x, self.y
        extend: make x_range wider by half of delta_x on left and right
        """
        self.sca()
        if extend:
            x_range = [x_range[0]-delta_x/2, x_range[1]+delta_x/2]
        n_bins = round((x_range[1] - x_range[0]) / delta_x)
        self.y_hist, self.x_hist, _ = plt.hist(data, bins=n_bins, range=x_range,
                                               density=density, edgecolor=edgecolor, linewidth=linewidth)
        self.x_hist = np.array([self.x_hist[i] for i in range(n_bins)]) + delta_x/2

    def legend(self):
        self.sca()
        plt.legend()

    def title(self, label, loc=None, **kwargs):
        self.sca()
        plt.title(label, loc=loc, **kwargs)

    def text(self, x, y, s, **kwargs):
        self.sca()
        plt.text(x, y, s, **kwargs)

    def xlabel(self, xlabel, loc=None, **kwargs):
        self.sca()
        plt.xlabel(xlabel, loc=loc, **kwargs)

    def ylabel(self, ylabel, loc=None, **kwargs):
        self.sca()
        plt.ylabel(ylabel, loc=loc, **kwargs)


class Subplots:

    def __init__(self, nrows=1, ncols=1, title=None, fontsize='x-large', **kwargs):
        self.fig, self.axs = plt.subplots(nrows, ncols, **kwargs)
        if hasattr(self.axs, 'shape'):
            self.shape = self.axs.shape
        else:
            self.shape = []
        self.plots = None
        self.title = title
        self.fontsize = fontsize

    def load_plots(self):
        if len(self.shape) == 2:
            self.plots = [[Plot(j) for j in i] for i in self.axs]
        elif len(self.shape) == 1:
            self.plots = [Plot(i) for i in self.axs]
        elif len(self.shape) == 0:
            self.plots = [Plot(self.axs)]

    def __enter__(self):
        plt.suptitle(self.title, fontsize=self.fontsize)
        self.load_plots()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show()

    def __getitem__(self, item):
        assert self.plots is not None
        return self.plots[item]

    def __call__(self, *args) -> Plot:
        if len(args) == 2:
            i, j = args
            return self[i][j]
        elif len(args) == 1:
            i, = args
            return self[i]
        elif len(args) == 0:
            return self[0]
        else:
            raise ValueError('Too many indices')


# -------------------------------- EXAMPLES --------------------------------


def test_single():
    p = Plot()
    p.scatter(x=[1, 2, 3], y=[2, 3, 3])
    p.plot(x=[0, 4], y=[2, 3])
    plt.title('Title')
    plt.show()


def test_hist():
    p = Plot()
    p.title('Hist')
    p.xlabel('xlabel', loc='right')
    p.ylabel('ylabel', loc='top')
    p.text(0, 0, 'origin')
    p.hist(data=[1, 2, 2, 3, 3], x_range=[.5, 3.5], delta_x=1)
    print(p.x_hist)
    print(p.y_hist)
    plt.show()


def test_multiple():
    fig, ax = plt.subplots(nrows=2)

    p1 = Plot(ax[0])
    p1.scatter(x=[1, 2, 3], y=[2, 3, 3])
    p1.plot(x=extended(p1.x_scatter, extend_ratio=.3), y=[2.5 for _ in range(5)])
    p1.errorbar(p1.x_scatter, p1.y_scatter, xerr=[.1, .1, .1], yerr=[.1, .2, .3])
    p1.legend()

    p2 = Plot(ax[1])
    p2.scatter(x=[0, 1, 2, 3], y=[0, 1, 4, 9])
    p2.plot(x=list(range(4)), func=lambda x: x ** 2)

    plt.show()


def test_subplots():
    with Subplots(nrows=2, ncols=2, title='Title') as plots:
        plots(0, 0).title('subtitle')
        plots(0, 0).scatter(x=[1, 2], y=[2, 3])
        plots(0, 0).plot(x=extended(plots(0, 0).x_scatter, extend_ratio=.3), func=lambda x: x + 1)

        plots(0, 1).hist(data=np.random.random(100) * 3, x_range=[0, 3], delta_x=1.)
        plots(0, 1).scatter(list(range(4)), [.2, .2, .2, .2], c='orange')
        plots(0, 1).plot(x=[0, 3], func=lambda x: 1/3)


if __name__ == '__main__':
    test_single()
    test_hist()
    test_multiple()
    test_subplots()
