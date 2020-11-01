import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ResNet50Backbone(nn.Module):
    """Return (BATCH_SIZE, 512, 16, 128)"""

    def __init__(self, pretrained_resnet=True):
        super(ResNet50Backbone, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=pretrained_resnet)
        layers_list = list(resnet50.children())
        self.conv_1 = layers_list[0]
        self.batch_norm_1 = layers_list[1]
        self.relu = layers_list[2]
        self.max_pool_1 = layers_list[3]
        self.first_block = layers_list[4]
        self.second_block = layers_list[5]

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(self.batch_norm_1(x))
        x = self.max_pool_1(x)
        x = self.first_block(x)
        x = self.second_block(x)
        return x