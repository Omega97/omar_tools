"""
Generators that computes average and standard deviation as data
comes in, without the need to locally store the whole data-set.
"""


def _update_fun(old, new_x, weight):
    return (old * weight + new_x) / (weight + 1)


def average(x_gen):
    avg = 0.
    for n, x in enumerate(x_gen):
        avg = _update_fun(avg, x, n)
        yield avg


def avg_and_std(x_gen):
    avg = var = 0.

    for n, x in enumerate(x_gen):
        old_avg = avg
        avg = _update_fun(avg, x, n)
        var = _update_fun(var + (avg - old_avg) ** 2, (x - avg) ** 2, n)
        yield avg, var ** .5


def covariance(x_gen, y_gen):
    avg_x = avg_y = avg_p = 0.

    for n, (x, y) in enumerate(zip(x_gen, y_gen)):
        avg_x = _update_fun(avg_x, x, n)
        avg_y = _update_fun(avg_y, y, n)
        avg_p = _update_fun(avg_p, x * y, n)
        yield avg_p - avg_x * avg_y
