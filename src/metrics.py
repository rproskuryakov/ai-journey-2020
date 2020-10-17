from abc import (
    ABC,
    abstractmethod,
)
from typing import List

import editdistance


class StringMetric(ABC):
    def __init__(self):
        self.numerator = 0
        self.denominator = 0
        self.name = None

    @abstractmethod
    def __call__(self, *, ground_strings: List[str], pred_strings: List[str]):
        pass

    def calculate(self):
        assert self.denominator != 0, "Metric cannot be calculated, division by zero"
        metric = self.numerator / self.denominator
        self.numerator, self.denominator = 0, 0
        return metric

    @abstractmethod
    def name(self):
        pass


class CharacterErrorRate(StringMetric):
    def __call__(self, *, ground_strings: List[str], pred_strings: List[str]):
        self.denominator += sum(len(string) for string in ground_strings)
        for true_str, pred_str in zip(ground_strings, pred_strings):
            self.numerator += editdistance.eval(pred_str, true_str)

    def name(self):
        return "CharacterErrorRate"


class WordErrorRate(StringMetric):
    def __call__(self, *, ground_strings: List[str], pred_strings: List[str]):
        self.denominator += sum(len(string.split()) for string in ground_strings)
        for true_str, pred_str in zip(ground_strings, pred_strings):
            self.numerator += editdistance.eval(pred_str.split(), true_str.split())

    def name(self):
        return "WordErrorRate"


class StringAccuracy(StringMetric):
    def __call__(self, *, ground_strings: List[str], pred_strings: List[str]):
        self.denominator += len(ground_strings)
        for true_str, pred_str in zip(ground_strings, pred_strings):
            self.numerator += int(true_str == pred_str)

    def name(self):
        return "StringAccuracy"
