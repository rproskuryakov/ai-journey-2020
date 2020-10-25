import copy
from typing import Dict


class MetricAccumulator:
    def __init__(self):
        self._name_to_metric = {}

    def accumulate(self, metric, name):
        val = self._name_to_metric.get(name, 0)
        self._name_to_metric[name] = val + metric

    @property
    def name_to_metric(self) -> Dict[str, float]:
        return copy.deepcopy(self._name_to_metric)
