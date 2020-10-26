from abc import ABC
from abc import abstractmethod
from typing import List

import torch


class BaseDecoder(ABC):

    @abstractmethod
    def __init__(self, letters: List[str], *args, **kwargs):
        self.letters = letters
        self.blank_id = kwargs.get("blank_id", 0)
        self.collapse_repeated = kwargs.get("collapse_repeated", True)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Return 2D-array (BATCH_SIZE, N_TIMESTEPS)"""
        pass

    def decode(self, decoder_output: torch.Tensor, labels: torch.Tensor = None, label_lengths: torch.Tensor = None):
        decodes = []
        targets = []
        for i, args in enumerate(decoder_output):
            decode = []
            if (labels and label_lengths):
                targets.append(self.int_to_text(labels[i][:label_lengths[i]].tolist()))
            for j, index in enumerate(args):
                if index != self.blank_id:
                    if self.collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.int_to_text(decode))
        return decodes, targets

    def int_to_text(self, indexes):
        decoded = []
        for i in indexes:
            try:
                decoded.append(self.letters[int(i)])
            except IndexError:
                print(indexes)
                print(i)
        return "".join(decoded)

