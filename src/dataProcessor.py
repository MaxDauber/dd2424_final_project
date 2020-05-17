import numpy as np
from collections import Counter

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    inputs = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    for i in range(batch_size):
        for u in range(seq_len):
            inputs[i, u, sequence[i * seq_len + u]] = 1
    return inputs

class DataCharProcessor:
    def __init__(self, filename):
        self.text = []
        self.char2int = {}
        self.int2char = {}
        self.n_vocab = 0
        self.loadData(filename)

    def loadData(self, file_name):
        with open(file_name) as file:
            self.text = file.read()

        chars = set(self.text)

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        self.n_vocab = len(self.char2int)

    def encodeData(self, seq_length, batch_size):
        # Creating lists that will hold our input and target sequences
        inputs_seq = []
        targets_seq = []
        e = 0
        dict_size = len(self.char2int)
        while e <= len(self.text) - seq_length * batch_size - 1:
            input_seq = [self.char2int[character] for character in self.text[e: e + seq_length * batch_size]]
            inputs_seq.append(one_hot_encode(input_seq, dict_size, seq_length, batch_size))

            target_seq = [self.char2int[character] for character in self.text[e + 1: e + 1 + seq_length * batch_size]]
            targets_seq.append(np.reshape(target_seq, (batch_size, -1)))

            e += seq_length * batch_size

        return inputs_seq, targets_seq

class DataWordProcessor:
    def __init__(self, filename):
        self.text = []
        self.char2int = {}
        self.int2char = {}
        self.n_words = 0
        self.loadData(filename)

    def loadData(self, file_name):
        with open(file_name) as file:
            self.text = file.read()
        self.text = self.text.split()

        word_counts = Counter(self.text)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int2char = {k: w for k, w in enumerate(sorted_vocab)}
        self.char2int = {w: k for k, w in self.int2char.items()}
        self.n_vocab = len(self.int2char)

        print('Vocabulary size', self.n_vocab)

    def encodeData(self, seq_len, batch_size):
        int_text = [self.char2int[w] for w in self.text]
        num_batches = int(len(int_text) / (seq_len * batch_size))
        in_text = int_text[:num_batches * batch_size * seq_len]
        out_text = int_text[1:num_batches * batch_size * seq_len+1]

        in_text = np.reshape(in_text, (batch_size, -1))
        out_text = np.reshape(out_text, (batch_size, -1))

        inputs = []
        targets = []
        for i in range(0, num_batches * seq_len, seq_len):
             inputs.append(in_text[:, i:i + seq_len])
             targets.append(out_text[:, i:i + seq_len])
        return inputs, targets