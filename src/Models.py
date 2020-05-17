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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states