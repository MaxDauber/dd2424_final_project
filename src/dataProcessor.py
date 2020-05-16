import numpy as np

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    inputs = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    for i in range(batch_size):
        for u in range(seq_len):
            inputs[i, u, sequence[i * seq_len + u]] = 1
    return inputs

class DataProcessor:
    def __init__(self, filename):
        self.text = []
        self.char2int = {}
        self.int2char = {}
        self.encoding_size = 0
        self.loadData(filename)

    def loadData(self, file_name):
        with open(file_name) as file:
            self.text = file.read()

        chars = set(self.text)

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        self.encoding_size = len(self.char2int)

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