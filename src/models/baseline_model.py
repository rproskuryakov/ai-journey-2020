import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BaselineNetwork"]


class BaselineNetwork(nn.Module):

    def __init__(self, n_letters):
        super(BaselineNetwork, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(4, 2), stride=2)
        self.conv_2 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(4, 2), stride=2)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool_4 = nn.MaxPool2d(kernel_size=(4, 1))
        self.conv_5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm5 = nn.BatchNorm2d(512)
        self.conv_6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool_6 = nn.MaxPool2d(kernel_size=(4, 1))
        self.conv_7 = nn.Conv2d(512, 512, kernel_size=(2, 2))
        self.blstm_1 = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.dropout_1 = nn.Dropout(0.2)
        self.blstm_2 = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.dropout_2 = nn.Dropout(0.2)
        self.dense = nn.Linear(256, n_letters)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = F.relu(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.pad(x, [0, 0, 3, 0])
        x = self.pool_4(x)
        x = F.relu(self.conv_5(x))
        x = self.batch_norm5(x)
        x = F.relu(self.conv_6(x))
        x = self.batch_norm6(x)
        x = F.pad(x, [0, 0, 3, 0])
        x = self.pool_6(x)
        x = self.conv_7(x)
        x = x.squeeze(dim=2).permute(0, 2, 1)
        x, _ = self.blstm_1(x)
        x = self.dropout_1(x)
        x, _ = self.blstm_2(x)
        x = self.dropout_2(x)
        x = self.dense(x)
        x = F.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)
