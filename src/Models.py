import torch
from torch import nn
from torch.nn import functional
import torch.utils.data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, input_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))

        out, hidden = self.rnn(x, hidden)

        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(BasicLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True)
        self.Wo = nn.Linear(hidden_dim, input_size)

    def forward(self, X, H=None):
        if H is None:
            H = self.init_hidden(X.size(0))
        X, H = self.lstm(X, H)
        out = self.Wo(X)
        return functional.log_softmax(out, 2), H

    def init_hidden(self, batch_size):
        h = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        c = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        return (h, c)

class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(StackedLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
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

        self.Wo = nn.Linear(hidden_dim*3, input_size)

    def forward(self, X, H=None):
        if H is None:
            H = self.init_hidden(X.size(0))
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