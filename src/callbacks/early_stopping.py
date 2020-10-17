from typing import Union

import torch

from src.callbacks.base_callback import BaseCallback
from src.callbacks.utils import init_is_better


class EarlyStopping(BaseCallback):
    def __init__(
            self,
            mode: str = 'min',
            min_delta: float = 0.,
            patience: int = 10,
            percentage: bool = False,
    ):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.is_better = init_is_better(mode, min_delta, percentage)
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def __call__(self, loss: float) -> bool:
        if self.best is None:
            self.best = loss
            return False

        if self.is_better(loss, self.best):
            self.num_bad_epochs = 0
            self.best = loss
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False
