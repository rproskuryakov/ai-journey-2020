import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchnlp.nn import WeightDropGRU

__all__ = ["ResNet18AttentionNetwork"]

class _ResNet18Backbone(nn.Module):
    """Return (BATCH_SIZE, 64, 32, 256)"""

    def __init__(self, pretrained_resnet=True):
        super(_ResNet18Backbone, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=pretrained_resnet)
        layers_list = list(resnet18.children())
        self.conv_1 = layers_list[0]
        self.batch_norm_1 = layers_list[1]
        self.relu = layers_list[2]
        self.max_pool_1 = layers_list[3]
        self.first_block = layers_list[4]
        self.second_block = layers_list[5]
        self.third_block = layers_list[6]

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(self.batch_norm_1(x))
        x = self.max_pool_1(x)
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)
        return x


class ResNet18AttentionNetwork(nn.Module):
    """Return tensor with shape (256, 1, n_letters)"""

    def __init__(self, n_letters, pretrained_resnet=True):
        super(ResNet18AttentionNetwork, self).__init__()
        self.resnet_extractor = _ResNet18Backbone(pretrained_resnet=pretrained_resnet)
        # first gru block out of 2 parallel
        self.self_attention_1_y = nn.MultiheadAttention(512, 4)
        self.blstm_1_y = nn.GRU(512, 128, bidirectional=True, batch_first=True)#, weight_dropout=0.2)
        self.self_attention_2_y = nn.MultiheadAttention(256, 4)
        self.blstm_2_y = nn.GRU(256, 128, bidirectional=True, batch_first=True)#, weight_dropout=0.2)

        # second gru-block out of 2 parallel
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.self_attention_1_z = nn.MultiheadAttention(128, 4)
        self.blstm_1_z = nn.GRU(128, 64, bidirectional=True, batch_first=True)#, weight_dropout=0.2)
        self.self_attention_2_z = nn.MultiheadAttention(128, 4)
        self.blstm_2_z = nn.GRU(128, 64, bidirectional=True, batch_first=True)#, weight_dropout=0.2)

        self.dense = nn.Linear(384, n_letters)

    def forward(self, x):
        x = self.resnet_extractor(x.float())

        # first block out of two parallels
        batch_size, n_channels, *_ = x.size()
        y = x.reshape(batch_size, n_channels, -1)
        y, _ = self.self_attention_1_y(y, y, y)
        y, _ = self.blstm_1_y(y)
        y, _ = self.self_attention_2_y(y, y, y)
        y, _ = self.blstm_2_y(y)

        # second block out of two parallels
        z = self.max_pool(x)
        batch_size, n_channels, *_ = z.size()
        z = z.reshape(batch_size, n_channels, -1)
        z, _ = self.self_attention_1_z(z, z, z)
        z, _ = self.blstm_1_z(z)
        z, _ = self.self_attention_2_z(z, z, z)
        z, _ = self.blstm_2_z(z)

        # concat and return probas
        x = torch.cat((y, z), 2)
        x = self.dense(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)
