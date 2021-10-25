import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

class TextCNNConfig:
    def __init__(self, n_vocab, n_embed, num_classes, embedding_pretrained, num_filters, filter_sizes=[3, 4, 5], dropout=0.1):
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.num_classes = num_classes
        self.embedding_pretrained = embedding_pretrained
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.n_embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.n_embed)) for k in config.filter_sizes])  # in_channel, out_channel, H, W
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = torch.relu(conv(x)).squeeze(3)
        x = nn.functional.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out