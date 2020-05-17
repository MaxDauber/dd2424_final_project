import torch
from torch import nn
import numpy as np
from Models import BasicLSTM, RNN, StackedLSTM
from dataProcessor import DataCharProcessor, DataWordProcessor, one_hot_encode
import matplotlib.pyplot as plt

def detach(layers):
    if (type(layers) is list) or (type(layers) is tuple):
        for l in layers:
            detach(l)
    else:
        layers.detach_()

def predict(model, characters, dataProcessor):

    characters = [dataProcessor.char2int[c] for c in characters]
    characters = one_hot_encode(characters, dataProcessor.n_vocab, len(characters), 1)
    characters = torch.from_numpy(characters)

    out, hidden = model(characters)
    char_id = torch.multinomial(
        out[0][-1].exp(), num_samples=1).item()

    return dataProcessor.int2char[char_id], hidden

def sample(model, out_len, start, dataProcessor):
    model.eval() # eval mode

    chars = [ch for ch in start]
    size = out_len - len(chars)

    for ii in range(size):
        char, h = predict(model, chars, dataProcessor)
        chars.append(char)

    return ''.join(chars)

def predictW(model, characters, dataProcessor):
    characters = [dataProcessor.char2int[c] for c in characters]
    characters = torch.from_numpy(np.reshape(characters, (1, -1)))

    out, hidden = model(characters)
    char_id = torch.multinomial(
        out[0][-1].exp(), num_samples=1).item()

    return dataProcessor.int2char[char_id], hidden

def sampleW(model, out_len, start, dataProcessor):
    model.eval() # eval mode

    words = [ch for ch in start]
    size = out_len - len(words)

    for ii in range(size):
        char, h = predictW(model, words, dataProcessor)
        words.append(char)

    return ' '.join(words)

def train(model, inputs_seq, targets_seq, n_epochs, eta, dataProcessor):
    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=eta)

    smooth_loss = None
    iters = []
    losses = []

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
            loss = criterion(output.transpose(2, 1), targets.long())
            loss.backward()  # Does backpropagation and calculates gradients

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()  # Updates the weights accordingly

            smooth_loss = loss.item() if smooth_loss is None else .999 * smooth_loss + 0.001 * loss.item()

            if iteration-1 % 100:
                losses.append(smooth_loss)
                iters.append(iteration)
            if iteration % 100 == 0:
                print('Iteration {}, Epoch: {}/{}.............'.format(iteration, epoch, n_epochs), end=' ')
                print("Smooth loss : {} Loss: {:.4f}".format(smooth_loss, loss.item()))
                if words:
                    print(sampleW(model, 100, ['you'], dataProcessor))
                else:
                    print(sample(model, 200, "a", dataProcessor))

    plt.plot(iters, losses)
    plt.xlabel('Update')
    plt.ylabel('loss')
    plt.title("Graph of smooth loss at every 100th update step")
    plt.show()


if __name__ == "__main__":

    # Choose to generate with words or characters
    words = True

    # Define the parameters used for the data
    file_name = "../Datasets/tree_bank_train.txt"
    seq_length = 32
    batch_size = 16
    embedding_size = 64 if words else None
    # Define the parameters used for the model
    n_layers = 1
    hidden_size = 64
    # Define the parameters used for the training
    eta = 0.01
    n_epoch = 5

    processor = DataWordProcessor(file_name) if words else DataCharProcessor(file_name)
    inputs, targets = processor.encodeData(seq_length, batch_size)

    # Create the model
    model = StackedLSTM(input_size=processor.n_vocab, hidden_dim=hidden_size, n_layers=n_layers, embedding_size=embedding_size)

    train(model, inputs, targets, n_epoch, eta, processor)
