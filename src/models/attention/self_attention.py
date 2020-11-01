import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SelfAttention"]


class SelfAttention(nn.Module):
    """Input size (BATCH_SIZE, INPUT_LENGTH, EMB_SIZE)"""

    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, x):
        scores = F.softmax(torch.bmm(x, torch.transpose(x, 1, 2)), dim=1)
        return torch.bmm(scores, x)
