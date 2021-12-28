"""
This module makes pyplot easier to use
"""
__author__ = "Omar Cusma Fait"
__date__ = (19, 12, 2021)
__version__ = "1.0.0"

import matplotlib.pyplot as plt
import numpy as np


def extended(array, extend_ratio=None, x_min=None, x_max=None):
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
        self.x_hist_center = None
        self.y_hist = None
        self._plot = None
        self._scatter = None
        self._errorbar = None

    def sca(self):
        if self.ax is not None:
            plt.sca(self.ax)

    def scatter(self, x, y, marker='+', color='k', s=50, label='data', zorder=2, **kwargs):
        self.sca()
        self.x_scatter = np.array(x)
        self.y_scatter = np.array(y)
        self._scatter = plt.scatter(x, y, marker=marker, color=color, s=s, label=label, zorder=zorder, **kwargs)

    def plot(self, x, y=None, func=None, linestyle='--', label='fit', **kwargs):
        """used to make fit"""
        self.sca()
        if y is None:
            assert func
            y = [func(i) for i in x]
        self.x_plot = np.array(x)
        self.y_plot = np.array(y)
        self._plot = plt.plot(x, y, linestyle=linestyle, label=label, **kwargs)

    def errorbar(self, x, y, xerr=None, yerr=None, color='k', fmt='none',
                 solid_capstyle='projecting', capsize=3, label='errors', **kwargs):
        self.sca()
        self._errorbar = plt.errorbar(x, y, yerr=yerr, xerr=xerr, color=color, fmt=fmt, solid_capstyle=solid_capstyle,
                                      capsize=capsize, label=label, **kwargs)

    def hist(self, x, x_range=None, n_bins=None, delta_x=None,
             linewidth=1.2, density=True, edgecolor='black', **kwargs):
        """
        - plot histogram
        - set self.x, self.y
        extend: make x_range wider by half of delta_x on left and right
        """
        self.sca()

        if x_range is None:
            x_range = [min(x), max(x)]

        if delta_x is None and n_bins is not None:
            delta_x = (x_range[-1] - x_range[0]) / n_bins

        if n_bins is None and delta_x is not None:
            n_bins = round((x_range[1] - x_range[0]) / delta_x)

        self.y_hist, self.x_hist, _ = plt.hist(x, bins=n_bins, range=x_range, density=density,
                                               edgecolor=edgecolor, linewidth=linewidth, **kwargs)
        n_bins = len(self.x_hist) - 1
        delta_x = (self.x_hist[-1] - self.x_hist[0]) / n_bins
        self.x_hist_center = np.array([self.x_hist[i] for i in range(n_bins)]) + delta_x/2

    def hist_prob(self, func):
        """compute the area of the function for each bin"""
        assert self.x_hist is not None
        assert self.x_hist_center is not None
        n_bins = len(self.x_hist_center)
        x0 = self.x_hist[0]
        x1 = self.x_hist[-1]
        delta_x = (x1 - x0) / n_bins
        p = np.array([func(self.x_hist[i]) + 4 * func(self.x_hist_center[i]) + func(self.x_hist[i+1])
                      for i in range(n_bins)]) / 6 * delta_x
        return p

    def legend(self):
        self.sca()
        plt.legend()

    def title(self, label, **kwargs):
        self.sca()
        plt.title(label, **kwargs)

    def text(self, x, y, s, **kwargs):
        self.sca()
        plt.text(x, y, s, **kwargs)

    def xlabel(self, xlabel, **kwargs):
        self.sca()
        plt.xlabel(xlabel, **kwargs)

    def ylabel(self, ylabel, **kwargs):
        self.sca()
        plt.ylabel(ylabel, **kwargs)

    def set_xlim(self, x0, x1, **kwargs):
        self.sca()
        self.ax.set_xlim(x0, x1, **kwargs)

    def set_ylim(self, y0, y1, **kwargs):
        self.sca()
        self.ax.set_ylim(y0, y1, **kwargs)


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

    def get_plots(self):
        return self.plots

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
    p.scatter(x=[1, 2, 3], y=[2, 3, 3], s=50)
    p.plot(x=[0, 4], y=[2, 3])
    plt.title('Title')
    plt.show()


def test_hist():
    p = Plot()
    p.title('Hist')
    p.xlabel('xlabel')
    p.ylabel('ylabel')
    p.text(0, 0, 'origin')
    p.hist(x=np.random.normal(size=200), label='data')
    plt.legend()
    print('x =', p.x_hist)
    print('y =', p.y_hist)
    print('sum(y) =', sum(p.y_hist))
    v = p.hist_prob(func=lambda x: np.exp(-x**2/2) / (2 * np.pi)**.5)
    print('p =', v)
    print('sum(p) =', sum(v))
    p.scatter(p.x_hist_center, v, color='orange', marker='+')
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

        [[a, b], [c, d]] = plots.get_plots()

        a.title('subtitle')
        a.scatter(x=[1, 2], y=[2, 3])
        a.plot(x=extended(plots(0, 0).x_scatter, extend_ratio=.3), func=lambda x: x + 1)

        b.hist(x=np.random.random(100) * 3)
        b.scatter(list(range(4)), [.2, .2, .2, .2], color='orange')
        b.plot(x=[0, 3], func=lambda x: 1/3)
        b.set_xlim(-1, 4)
        b.set_ylim(0, 1)

        c.hist(np.random.random(size=50))

        d.hist(np.random.normal(size=50))


if __name__ == '__main__':
    test_single()
    test_hist()
    test_multiple()
    test_subplots()
