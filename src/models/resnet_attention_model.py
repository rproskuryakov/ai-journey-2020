import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn import WeightDropGRU

from .backbone import ResNet18Backbone

__all__ = ["ResNet18AttentionNetwork"]


class ResNet18AttentionNetwork(nn.Module):
    """Return tensor with shape (256, 1, n_letters)"""

    def __init__(self, n_letters, pretrained_resnet=True):
        super(ResNet18AttentionNetwork, self).__init__()
        self.resnet_extractor = ResNet18Backbone(pretrained_resnet=pretrained_resnet)

        # first conv-block out of 2 parallel
        self.y_conv_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 1), stride=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 1), stride=0),
            nn.ReLU(inplace=True),
        )

        # first gru block out of 2 parallel
        self.self_attention_1_y = nn.MultiheadAttention(64, 4)
        self.blstm_1_y = WeightDropGRU(64, 128, bidirectional=True, batch_first=True, weight_dropout=0.2)
        self.self_attention_2_y = nn.MultiheadAttention(256, 4)
        self.blstm_2_y = WeightDropGRU(256, 128, bidirectional=True, batch_first=True, weight_dropout=0.2)

        # second conv-block out of 2 parallel
        self.z_conv_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=0),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 1), stride=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(4, 1), stride=0),
            nn.ReLU(inplace=True)
        )

        # second gru-block out of 2 parallel
        self.self_attention_1_z = nn.MultiheadAttention(64, 4)
        self.blstm_1_z = WeightDropGRU(64, 64, bidirectional=True, batch_first=True, weight_dropout=0.2)
        self.self_attention_2_z = nn.MultiheadAttention(128, 4)
        self.blstm_2_z = WeightDropGRU(128, 64, bidirectional=True, batch_first=True, weight_dropout=0.2)

        self.dense = nn.Linear(384, n_letters)

    def forward(self, x):
        x = self.resnet_extractor(x.float())

        # first block out of two parallels
        y = self.y_conv_block(x)
        y = y.squeeze(dim=2).permute(0, 2, 1)
        y, _ = self.self_attention_1_y(y, y, y)
        y, _ = self.blstm_1_y(y)
        y, _ = self.self_attention_2_y(y, y, y)
        y, _ = self.blstm_2_y(y)

        # second block out of two parallels
        z = self.z_conv_block(x)
        z = z.squeeze(dim=2).permute(0, 2, 1)
        z, _ = self.self_attention_1_z(z, z, z)
        z, _ = self.blstm_1_z(z)
        z, _ = self.self_attention_2_z(z, z, z)
        z, _ = self.blstm_2_z(z)

        # concat and return probas
        x = torch.cat((y, z), 2)
        x = self.dense(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)
