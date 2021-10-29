import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

class RnnConfig:
    def __init__(self, n_vocab, n_embed, num_classes, embedding_pretrained, layer_num=1, layer_type='LSTM', hidden_size=256, use_attn=False, dropout=0.1, bidirectional=False):
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.num_classes = num_classes
        self.embedding_pretrained = embedding_pretrained

        assert layer_type in ['RNN', 'LSTM', 'GRU']
        if layer_type == 'RNN':
            self.layer_obj = nn.RNN
        elif layer_type == 'LSTM':
            self.layer_obj = nn.LSTM
        elif layer_type == 'GRU':
            self.layer_obj = nn.GRU
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout


class RnnModel(nn.Module):
    def __init__(self, config: RnnConfig):
        super(RnnModel, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.n_embed, padding_idx=config.n_vocab - 1)
        self.rnn = config.layer_obj(config.n_embed, config.hidden_size,
                                    num_layers=config.layer_num,
                                    bidirectional=config.bidirectional,
                                    batch_first=True, dropout=config.dropout)
        if config.bidirectional:
            self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        else:
            self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.rnn(x)[0]
        out = self.fc(x[:, -1])
        return out

if __name__ == '__main__':
    config = RnnConfig(100, 128, 8, None, layer_num=2, bidirectional=True)
    rnn = RnnModel(config)
    x = torch.tensor([[1, 3], [2, 6]])
    print(rnn(x))