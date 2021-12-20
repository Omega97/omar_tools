"""
                    Debug

Add the @Debug decorator on top of any method.
This will show you:
- where it was called
- it's inputs
- where it returned
- it's output
- the nested structure of all calls
"""


def debug_print(s='', indent=0):
    print(Debug.separator * (Debug.count + indent) + ' ' + s)


class Debug:

    count = 0
    separator = ' .  '

    def __init__(self, fun):
        self.fun = fun
        self.out = None

    def _pre_print(self, *args, **kwargs):
        s = f'{self.fun.__name__}('
        if len(args):
            s += f'{", ".join([str(i) for i in args])}'
        if len(kwargs):
            s += f', {", ".join([str(i) + "=" + str(kwargs[i]) for i in kwargs])}'
        s += ')'
        debug_print()
        debug_print(s)
        Debug.count += 1

    def _post_print(self):
        s = f'{self.out}'
        Debug.count -= 1
        debug_print(indent=1)
        debug_print(s)

    def __call__(self, *args, **kwargs):
        self._pre_print(*args, **kwargs)
        self.out = self.fun(*args, **kwargs)
        self._post_print()
        return self.out
