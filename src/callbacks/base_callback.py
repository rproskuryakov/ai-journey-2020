from abc import ABC
from abc import abstractmethod


class BaseCallback(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
