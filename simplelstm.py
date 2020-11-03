import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    Class for simple lstm used for text classification
    """
    def __init__(self, vocab_size, hidden_dim=100, emb_dim=300, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim,
                               hidden_dim,
                               num_layers=1,
                               dropout=dropout)
        self.linear_layers = []
        self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Dropout(dropout))
        self.linear_layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 4))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Dropout(dropout))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim // 4, 5)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds
