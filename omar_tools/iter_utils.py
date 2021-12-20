"""         iter utils

Common tools to handle iterator
"""
__author__ = "Omar Cusma Fait"
__date__ = (21, 8, 2021)
__version__ = '1.5.4'

import os
from time import time
from random import random
from itertools import tee, filterfalse
from collections import Counter


# ----------------------------------------------------------------
# ----------------------- frequently used ------------------------
# ----------------------------------------------------------------


def n_wise(n):
    """iterate n elements at the time, each time drop first and add new element at the end"""
    def wrap(itr):
        iterators = tee(itr, n)
        for i in range(n):
            for _ in range(i):
                next(iterators[i])
        return zip(*iterators)
    return wrap


def i_print(itr, n=None, head='\n'):
    """print an iterable (first n elements) """
    print(end=head)
    for i, x in enumerate(itr):
        if i == n:
            return
        print(x)


def d_print(dct, width=8, head='\n'):
    """print dictionary"""
    print(end=head)
    for i in dct:
        print(f'{i:>{width}} \t{dct[i]}')


def once_every_n(n):
    """yields only one element every n"""
    def wrap(itr):
        for i, x in enumerate(itr):
            if i % n == 0:
                yield x
    return wrap


def skip_n(n):
    """skip n elements of itr """
    def wrap(itr):
        for i, x in enumerate(itr):
            if i >= n:
                yield x
    return wrap


def group_by_n(n):
    """group elements of itr in n-long tuples"""
    def wrap(itr):
        c = tuple()
        for i, x in enumerate(itr):
            c += (i, )
            if i % n == n - 1:
                yield c
                c = tuple()
    return wrap


def stop_iter(stop_criterion):
    """
    stop_criterion: takes index and element as input, return True to stop iteration
    """
    def break_iter_(itr):
        for i, x in enumerate(itr):
            if stop_criterion(i, x):
                break
            else:
                yield i
    return break_iter_


def exp_iter(itr):
    """iterable yields exponentially less frequently"""
    count = 0
    n = 0
    for i, e in enumerate(itr):
        if i == n:
            yield e
            count += 1
            n = 2 ** count - 1


class CallableCounter(Counter):
    """
    - uses Counter to count objects (during __init__)
    - you can also count by calling on individual keys
    - read_itr: counts items in iterable
    """

    def __call__(self, key):
        value = self[key] if key in self else 1
        self.update({key: value})

    def read_itr(self, itr):
        for i in itr:
            self(i)


# ----------------------------------------------------------------
# ---------------------------- files -----------------------------
# ----------------------------------------------------------------


def read_file(path, encoding='utf-8'):
    """read file line by line"""
    with open(path, encoding=encoding) as file:
        for line in file:
            yield line[:-1] if line[-1] == '\n' else line


def read_file_lines(path, encoding='utf-8'):
    """read file line by line (skip empty lines)"""
    return filterfalse(lambda x: x == '', read_file(path, encoding))


def gen_files(path, search_sub_dir=True):
    """generate all paths of files in path"""
    for root, dirs, files in os.walk(path):
        for i in files:
            do_yield = True if search_sub_dir else root == path
            if do_yield:
                yield root + '\\' + i


def gen_dir(path, search_sub_dir=True):
    """generate all names of directories in path"""
    for root, dirs, files in os.walk(path):
        for i in dirs:
            do_yield = True if search_sub_dir else root == path
            if do_yield:
                yield root + '\\' + i


# ----------------------------------------------------------------
# --------------------------- strings ----------------------------
# ----------------------------------------------------------------


def skip_starts_with(*args: (str,)):
    """skip all elements that start with an element in args"""
    def wrap(itr):
        for i in itr:
            for j in args:
                if i.startswith(j):
                    break
            else:
                yield i
    return wrap


def keep_starts_with(*args: (str,)):
    """keep only elements that start with an element in args"""
    def _skip_starts_with(itr):
        for i in itr:
            for j in args:
                if i.startswith(j):
                    yield i
                    break
    return _skip_starts_with


def skip_ends_with(*args: (str,)):
    """skip all elements that start with an element in args"""
    def wrap(itr):
        for i in itr:
            for j in args:
                if i.endswith(j):
                    break
            else:
                yield i
    return wrap


def keep_ends_with(*args: (str,)):
    """keep only elements that start with an element in args"""
    def _skip_starts_with(itr):
        for i in itr:
            for j in args:
                if i.endswith(j):
                    yield i
                    break
    return _skip_starts_with


# ----------------------------------------------------------------
# -------------------------- less used ---------------------------
# ----------------------------------------------------------------


def gen_apply(f):
    """apply f on elements of itr"""
    def _gen_apply(itr):
        """apply f on elements of itr"""
        for i in itr:
            yield f(i)
    return _gen_apply


def gen_best_score(itr, best_score=None):
    """ pick from itr the (score, *obj) with the best score"""
    for x in itr:
        new_score, *new_obj = x
        if True if best_score is None else new_score > best_score:
            best_score, best_obj = new_score, new_obj
            yield [best_score] + best_obj


def get_best_n(n):
    def _get_best_n(itr):
        """itr must yield objects with __getitem__ and len >= 1 like (score, other)"""
        store = [next(itr)]
        for i in itr:
            if i[0] >= store[-1][0]:
                store += [i]
                store.sort(reverse=True)
                if len(store) > n:
                    store = store[:n]
                yield tuple(store)
    return _get_best_n


def loop_range(*args, **kwargs):
    """iter in loop over range"""
    while True:
        for i in range(*args, **kwargs):
            yield i


def split_data(p):
    """
    splits elements of itr in 2 groups
    yields (0, element) with frequency p
    yields (1, element) with frequency 1-p
    :param p: proportion (frequency)
    """
    def _split_data(itr):
        for i in itr:
            yield (0 if random() < p else 1), i
    return _split_data


# ----------------------------------------------------------------
# ---------------------------- time ------------------------------
# ----------------------------------------------------------------


def yield_time():
    """yield time lapsed from call"""
    t0 = time()
    while True:
        yield time() - t0


def yield_periodically(itr, period):
    """yield periodically, only after "period" of time"""
    t = time()
    for i in itr:
        if time() - t > period:
            t += period
            yield i


def yield_timer(period):
    """yield each time a 'period' of time has passed"""
    return yield_periodically(yield_time(), period)


# ----------------------------------------------------------------
# --------------------------- special ----------------------------
# ----------------------------------------------------------------


def tensor_gen(shape):
    """yields tuples (a_0, ... a_n) of indices a_i in range(shape[i])"""
    if shape:
        for i in range(shape[0]):
            for j in tensor_gen(shape[1:]):
                yield (i, ) + j
    else:
        yield tuple()


def recursive_iter(*args):
    """iterate recursively over all args"""
    elements = [list(i) for i in args]
    shape = [len(i) for i in elements]
    for i in tensor_gen(shape):
        yield tuple(elements[j][i[j]] for j in range(len(elements)))


# ----------------------------------------------------------------
# ------------------------- decorators ---------------------------
# ----------------------------------------------------------------


def gen_next_n_decorator(n):
    """the decorated methods yield only n times (n > 0)"""
    def wrap1(fun):
        def wrap2(*a, **kw):
            j = n
            for i in fun(*a, **kw):
                yield i
                j -= 1
                if not j:
                    break
        return wrap2
    return wrap1


def one_in_n_decorator(n):
    """the decorated methods yield only once every n times"""
    def wrap1(fun):
        def wrap2(*a, **kw):
            j = n
            for i in fun(*a, **kw):
                j -= 1
                if not j:
                    yield i
                    j = n
        return wrap2
    return wrap1


# ----------------------------------------------------------------
# ------------------------- deprecated ---------------------------
# ----------------------------------------------------------------


def gen_next_n(n):  # islice(itr, n)
    """creates a generator of the next n elements of itr"""
    def wrap(itr):
        for i, x in enumerate(itr):
            if i == n:
                return
            yield x
    return wrap


def infinite_range(start=0, step=1):    # count(start, step)
    """like range, but never ends"""
    n = start
    while True:
        yield n
        n += step


def loop_iter(itr, n):  # cycle
    for i in tee(itr, n):
        for j in i:
            yield j


def loop_list(v):   # cycle
    """like range but loop instead of stopping at the end"""
    while True:
        for i in v:
            yield i


def filter_iter(criterion):     # filterfalse(predicate, iterable)
    """skip from iter the elements that satisfy the criterion"""
    def wrap(itr):
        for i in itr:
            if not criterion(i):
                yield i
    return wrap


def sum_gen(itr):   # accumulate
    """sum of generator"""
    s = 0
    for i in itr:
        s += i
        yield s
