import torch
from torch import nn
from Models import BasicLSTM
from dataProcessor import DataProcessor, one_hot_encode

def detach(layers):
    if (type(layers) is list) or (type(layers) is tuple):
        for l in layers:
            detach(l)
    else:
        layers.detach_()

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, characters, dataProcessor):
    # One-hot encoding our input to fit into the model
    characters = [dataProcessor.char2int[c] for c in characters]
    characters = one_hot_encode(characters, dataProcessor.encoding_size, len(characters), 1)
    characters = torch.from_numpy(characters)

    out, hidden = model(characters)
    char_id = torch.multinomial(
        out[0][-1].exp() / 100, num_samples=1).item()

    return dataProcessor.int2char[char_id], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start, dataProcessor):
    model.eval() # eval mode
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars, dataProcessor)
        chars.append(char)

    return ''.join(chars)

def train(model, inputs_seq, targets_seq, n_epochs, eta, dataProcessor):
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=eta)

    # Training Run
    for epoch in range(1, n_epochs + 1):
        hiddens = model.init_hidden(batch_size)
        for batch_idx in range(len(inputs_seq)):
            inputs = torch.from_numpy(inputs_seq[batch_idx])
            targets = torch.Tensor(targets_seq[batch_idx])
            detach(hiddens)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            output, hiddens = model(inputs, hiddens)
            loss = criterion(output.transpose(2, 1), targets.long())
            loss.backward()  # Does backpropagation and calculates gradients

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()  # Updates the weights accordingly
            if batch_idx % 10000 == 0:
                print('Iteration {}, Epoch: {}/{}.............'.format(batch_idx, epoch, n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                print(sample(model, 100, 'Har', dataProcessor))

if __name__ == "__main__":

    # Define the parameters used for the data
    file_name = "../Datasets/goblet_book.txt"
    seq_length = 25
    batch_size = 1
    # Define the paremeters used for the model
    n_layers = 1
    hidden_size = 100
    # Define the paremeters used for the training
    eta = 0.1
    n_epoch = 100

    processor = DataProcessor(file_name)
    inputs, targets = processor.encodeData(seq_length, batch_size)

    # Create the model
    model = BasicLSTM(input_size=processor.encoding_size, hidden_dim=hidden_size, n_layers=n_layers)

    train(model, inputs, targets, n_epoch, eta, processor)
