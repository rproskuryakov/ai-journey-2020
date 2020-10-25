import torch

from src.decoders.base_decoder import BaseDecoder


class GreedyDecoder(BaseDecoder):

    def __init__(self, letters, *args, **kwargs):
        super(GreedyDecoder, self).__init__(letters, *args, **kwargs)

    def __call__(self, output):
        return torch.argmax(output, dim=2).permute(1, 0)
