import copy
from abc import ABC
from abc import abstractmethod
from typing import List


class BaseDecoder(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def decode_labels(labels: List[int], letters: List[str]) -> List[str]:
        decode_labels = list()
        letters = copy.copy(letters)
        letters.append('')
        for sample in labels:
            decode_labels.append(''.join([letters[int(i)] for i in sample]))
        return decode_labels
