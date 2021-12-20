"""         Data handling
The File class:
1) Tries to load data from self.path (if path provided)
2) Tries to download data from web (if no data loaded but URL provided)
"""
__author__ = "Omar Cusma Fait"
__date__ = (20, 12, 2021)
__version__ = "1.0.0"

import pandas as pd


class Data:
    def __init__(self, data=None, path=None, url=None, load_method=pd.read_csv):
        self.data = data
        self.path = path
        self.url = url
        self.load_method = load_method
        if self.data is None:
            self._load_data()

    def __repr__(self):
        if self.data is None:
            return 'Data()'
        else:
            return str(self.get_data())

    def __getitem__(self, item):
        return self.data[item]

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def _load_data(self):
        if self.path is not None:
            try:
                self.set_data(self.load_method(self.path))
            except FileNotFoundError:
                pass
        elif self.url is not None:
            try:
                self.set_data(self.load_method(self.url))
            except FileNotFoundError:
                pass
            self.save()

    def save(self, path=None):
        if path is None:
            path = self.path
        self.get_data().to_csv(path, index=False)
