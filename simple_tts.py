import torch
import torch.nn as nn

class SimpleTTS(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 256)

        self.encoder = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )

        self.linear = nn.Linear(
            512,
            80
        )

    def forward(self, x):

        x = self.embedding(x)

        x, _ = self.encoder(x)

        x, _ = self.decoder(x)

        mel = self.linear(x)

        return mel