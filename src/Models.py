import torch
from torch import nn
from torch.nn import functional
import torch.utils.data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, embedding_size=None):
        super(RNN, self).__init__()

        self.embed = embedding_size is not None

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        if self.embed:
            self.embedding = nn.Embedding(input_size, embedding_size)
            self.rnn = nn.RNN(input_size=embedding_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, input_size)

    def forward(self, X, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(X.size(0))
        if self.embed:
            X = self.embedding(X.long())
        out, hidden = self.rnn(X, hidden)

        out = self.fc(out)

        return functional.log_softmax(out, 2), hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, embedding_size=None):
        super(BasicLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed = embedding_size is not None

        # Defining the layers
        if self.embed:
            self.embedding = nn.Embedding(input_size, embedding_size)
            self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

        self.Wo = nn.Linear(hidden_dim, input_size)

    def forward(self, X, H=None):
        if H is None:
            H = self.init_hidden(X.size(0))
        if self.embed:
            X = self.embedding(X.long())
        X, H = self.lstm(X, H)
        out = self.Wo(X)
        return functional.log_softmax(out, 2), H

    def init_hidden(self, batch_size):
        h = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        c = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        return (h, c)

class StackedLSTM(nn.Module):
    # This is a stacked LSTM model with skip connections
    def __init__(self, input_size, hidden_dim, n_layers, embedding_size=None):
        super(StackedLSTM, self).__init__()

        self.embed = embedding_size is not None
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        output_size = input_size
        if self.embed:
            self.embedding = nn.Embedding(input_size, embedding_size)
            input_size = embedding_size
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)
        self.lstm2 = nn.LSTM(
            input_size=input_size+hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)
        self.lstm3 = nn.LSTM(
            input_size=input_size+hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)

        self.Wo = nn.Linear(hidden_dim*3, output_size)

    def forward(self, X, H=None):
        if H is None:
            H = self.init_hidden(X.size(0))
        if self.embed:
            X = self.embedding(X.long())
        h1, h2, h3 = H
        outs_1, h1 = self.lstm1(X, h1)
        outs_2, h2 = self.lstm2(torch.cat((X, outs_1), 2), h2)
        outs_3, h3 = self.lstm3(torch.cat((X, outs_2), 2), h3)
        out = self.Wo(torch.cat((outs_1, outs_2, outs_3), 2))
        return functional.log_softmax(out, 2), [h1, h2, h3]

    def init_hidden(self, batch_size):
        hiddens = []
        for i in range(3):
            h = torch.randn(self.n_layers, batch_size, self.hidden_dim)
            c = torch.randn(self.n_layers, batch_size, self.hidden_dim)
            hiddens.append((h, c))
        return hiddens