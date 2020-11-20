import math

import torch
import torch.nn as nn
import torchvision.models

__all__ = ["ResNetTransformerNetwork"]


class _ResNet18Backbone(nn.Module):
    """Return (BATCH_SIZE, 256, 512), where 256 is INPUT_LENGTH and 512 is EMB_SIZE"""

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
        batch_size, n_channels, _, _ = x.size()
        return x.reshape(batch_size, n_channels, -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class ResNetTransformerNetwork(nn.Module):

    def __init__(self, n_letters, pretrained_resnet=True, batch_size=16):
        super(ResNetTransformerNetwork, self).__init__()
        self.resnet_extractor = _ResNet18Backbone(pretrained_resnet=pretrained_resnet)
        self.transformer = TransformerModel(ntoken=n_letters, ninp=512, nhead=16, nhid=512, nlayers=2)
        self.batch_size = 16

    def forward(self, x):
        features = self.resnet_extractor(x)
        mask = self.transformer.generate_square_subsequent_mask(self.batch_size)
        return self.transformer(features, mask)
