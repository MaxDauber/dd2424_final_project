import torch
from torch import nn
from torch.nn import functional
import torch.utils.data
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the parameters
        '''Weights'''
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

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            #print((i+1)*u)
            features[i, u, sequence[i*seq_len + u]] = 1
    return features

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, characters):
    # One-hot encoding our input to fit into the model
    characters = [char2int[c] for c in characters]
    characters = one_hot_encode(characters, dict_size, len(characters), 1)
    characters = torch.from_numpy(characters)

    out, hidden = model(characters)
    char_id = torch.multinomial(
        out[0][-1].exp() / 100, num_samples=1).item()

    return int2char[char_id], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

def detach(layers):
    '''
    Remove variables' parent node after each sequence,
    basically no where to propagate gradient
    '''
    if (type(layers) is list) or (type(layers) is tuple):
        for l in layers:
            detach(l)
    else:
        layers.detach_()  # layers = layers.detach()

book_fname = 'Datasets/goblet_book.txt'
with open(book_fname) as file:
    text = file.read()

chars = set(text)

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# Creating lists that will hold our input and target sequences
inputs_seq = []
targets_seq = []
e = 0
seq_length = 25
dict_size = len(char2int)
batch_size = 1
while e <= len(text)-seq_length*batch_size-1:

    input_seq = [char2int[character] for character in text[e: e + seq_length*batch_size]]
    inputs_seq.append(one_hot_encode(input_seq, dict_size, seq_length, batch_size))

    target_seq = [char2int[character] for character in text[e+1: e + 1 + seq_length*batch_size]]
    targets_seq.append(np.reshape(target_seq, (batch_size, -1)))

    e += seq_length*batch_size

# Instantiate the model with hyperparameters
model = RNN(input_size=dict_size, hidden_dim=50, n_layers=1)


# Define hyperparameters
n_epochs = 100
lr=0.1

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# Training Run
iteration = 0
for epoch in range(1, n_epochs + 1):
    hiddens = model.init_hidden(batch_size)
    for batch_idx in range(len(inputs_seq)):
        iteration += 1
        inputs = torch.from_numpy(inputs_seq[batch_idx])
        targets = torch.Tensor(targets_seq[batch_idx])
        detach(hiddens)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        output, hiddens = model(inputs, hiddens)
        loss = criterion(output.transpose(2,1), targets.long())
        loss.backward()  # Does backpropagation and calculates gradients

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()  # Updates the weights accordingly
        if iteration % 1000 == 0:
            print('Iteration {}, Epoch: {}/{}.............'.format(iteration, epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            print(sample(model, 100, 'Har'))