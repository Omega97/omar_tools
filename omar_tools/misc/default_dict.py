
class DefaultDict(dict):
    """
    dict-like object, returns default value if item not in self
    """
    def __init__(self, *args, default=0, **kwargs):
        self.default = default
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        else:
            return self.default


def test():
    d = DefaultDict(default=0)
    d['a'] = 1

    print(d)
    print(d['a'])
    print(d['b'])


if __name__ == '__main__':
    test()
