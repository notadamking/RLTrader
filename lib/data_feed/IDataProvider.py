from abc import ABCMeta, abstractmethod

class IDataProvider:
    __metaclass__ = ABCMeta

    @classmethod
    def get_data(self): raise NotImplementedError

