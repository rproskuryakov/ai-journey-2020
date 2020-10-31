import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention"]


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, K, V, Q):
        A = torch.bmm(K.transpose(1, 2), Q) / np.sqrt(Q.shape[1])
        A = F.softmax(A, 1)
        R = torch.bmm(V, A)
        return torch.cat((R, Q), dim=1)
