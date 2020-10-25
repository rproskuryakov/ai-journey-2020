import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ResNetBackbone(nn.Module):

    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        layers_list = list(resnet18.children())
        self.conv_1 = layers_list[0]
        self.batch_norm_1 = layers_list[1]
        self.relu = layers_list[2]
        self.max_pool_1 = layers_list[3]
        self.first_block = layers_list[4]

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(self.batch_norm_1(x))
        x = self.max_pool_1(x)
        x = self.first_block(x)
        return x


class ResNetNetwork(nn.Module):

    def __init__(self, n_letters, freeze_resnet=False):
        super(ResNetNetwork, self).__init__()
        self.resnet_extractor = ResNetBackbone()

        # first conv-block out of 2 parallel
        self.y_conv_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(4, 1), stride=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(4, 1), stride=0),
            nn.ReLU(inplace=True),

        )

        # first gru block out of 2 parallel
        self.blstm_1_y = nn.GRU(64, 128, bidirectional=True, batch_first=True)
        self.dropout_1_y = nn.Dropout(0.2)
        self.blstm_2_y = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.dropout_2_y = nn.Dropout(0.2)

        # second conv-block out of 2 parallel
        self.z_conv_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 1), stride=0),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(2, 1), stride=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(0, 1)),
            nn.MaxPool2d(kernel_size=(4, 1), stride=0),
        )

        # second gru-block out of 2 parallel
        self.blstm_1_z = nn.GRU(64, 64, bidirectional=True, batch_first=True)
        self.dropout_1_z = nn.Dropout(0.2)
        self.blstm_2_z = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.dropout_2_z = nn.Dropout(0.2)

        self.dense = nn.Linear(384, n_letters)

    def forward(self, x):
        x = self.resnet_extractor(x.float())

        # first block out of two parallels
        y = self.y_conv_block(x)
        y = y.squeeze(dim=2).permute(0, 2, 1)
        y, _ = self.blstm_1_y(y)
        y = self.dropout_1_y(y)
        y, _ = self.blstm_2_y(y)
        y = self.dropout_2_y(y)

        # second block out of two parallels
        z = self.z_conv_block(x)
        z = z.squeeze(dim=2).permute(0, 2, 1)
        z, _ = self.blstm_1_z(z)
        z = self.dropout_1_z(z)
        z, _ = self.blstm_2_z(z)
        z = self.dropout_2_z(z)

        # concat and return probas
        x = torch.cat((y, z), 2)
        x = self.dense(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)
