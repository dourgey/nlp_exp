import torch
import torch.nn as nn

class TextRCNNConfig:
    def __init__(self, n_vocab, n_embed, num_classes, embedding_pretrained, hidden_size, num_layers, dropout):
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.num_classes = num_classes
        self.embedding_pretrained = embedding_pretrained
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

class TextRCNN(nn.Module):
    def __init__(self, config: TextRCNNConfig):
        super(TextRCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.n_embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(input_size=config.n_embed, hidden_size=config.hidden_size, batch_first=True,
                            num_layers=config.num_layers, bidirectional=True, dropout=config.dropout)
        self.fc1 = nn.Linear(config.n_embed + 2 * config.hidden_size, 2 * config.hidden_size)
        self.fc2 = nn.Linear(2 * config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embed = self.embedding(x)
        # out = embed.unsqueeze(1)
        out, _ = self.lstm(embed)
        out = torch.cat((out, embed), 2)
        out = torch.tanh(self.fc1(out)).permute(0, 2, 1)
        out = torch.max_pool1d(out, out.shape[-1]).squeeze()
        return self.fc2(out)