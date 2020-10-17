from pathlib import Path
from typing import Optional
from typing import Union

import torch

from src.callbacks.base_callback import BaseCallback
from src.callbacks.utils import init_is_better


class SaveCheckpoints(BaseCallback):
    def __init__(
            self,
            model: torch.nn.Module,
            folder: Union[str, Path],
            only_best: bool = False,
            mode: str = 'min',
            percentage: bool = False,
            min_delta: float = 0.,
    ):
        self.only_best = only_best
        self.folder = Path(folder)
        self.model = model
        self.best_metric = None
        self.percentage = percentage
        self.is_better = init_is_better(mode, min_delta, percentage)

    def _save_checkpoint(self, name: Union[str, Path]):
        path = Path(self.folder) / name
        torch.save(self.model.cpu().state_dict(), path)
        self.model.cuda()

    def __call__(self, n_epoch: int, metric: Optional[Union[float, torch.Tensor]] = None) -> None:
        if not self.only_best:
            name = f'model_epoch_{n_epoch}.pth'
            self._save_checkpoint(name)
            return

        if self.only_best and metric is None:
            raise ValueError('metric must be passed')

        if self.best_metric is None:
            self.best_metric = metric
            name = f'best_model_epoch_{n_epoch}.pth'
            self._save_checkpoint(name)
            return

        if self.is_better(metric, self.best_metric):
            self.best_metric = metric
            name = f'best_model_epoch_{n_epoch}.pth'
            self._save_checkpoint(name)
