import torch
from torch import nn
from torch.nn import functional
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
        return [h, c]

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)

    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

book_fname = 'Datasets/goblet_book.txt'
with open(book_fname) as file:
    text = file.read()

chars = set(text)

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []
e = 0
seq_length = 25
while e <= len(text)-seq_length-1:
    # Remove last character for input sequence
    input_seq.append([char2int[character] for character in text[e: e + seq_length]])

    # Remove firsts character for target sequence
    target_seq.append([char2int[character] for character in text[e+1: e + 1 + seq_length]])
    e += seq_length + 1

dict_size = len(char2int)
batch_size = len(input_seq)

input_seq = one_hot_encode(input_seq, dict_size, seq_length, batch_size)
print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)
print(target_seq.view(-1).shape)

# Instantiate the model with hyperparameters
model = RNN(input_size=dict_size, hidden_dim=100, n_layers=1)


# Define hyperparameters
n_epochs = 100
lr=0.1

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward()  # Does backpropagation and calculates gradients

    nn.utils.clip_grad_norm(model.parameters, 1)
    optimizer.step()  # Updates the weights accordingly

    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        print(sample(model, 100, 'H'))