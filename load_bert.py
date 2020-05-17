import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import logging
import matplotlib.pyplot as plt


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print("print")
# model = BertModel.from_pretrained('bert-base-uncased')
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids)
# hidden_states = outputs[0]
# last_layer = hidden_states[-1]
# print(last_layer)


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
book_fname = 'Datasets/goblet_book.txt'
with open(book_fname) as file:
    text = file.read()


# text = "dummy. although he had already eaten a large meal, he was still very hungry."
target = "hungry"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = tokenized_text.index(target)
tokenized_text[masked_index] = '[MASK]'

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [1] * len(tokenized_text)

# this is for the dummy first sentence.
segments_ids[0] = 0
segments_ids[1] = 0

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

model.eval()


# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)


predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

print("Original:", text)
print("Masked:", " ".join(tokenized_text))

print("Predicted token:", predicted_token)
print("Other options:")
# just curious about what the next few options look like.
for i in range(10):
    predictions[0,masked_index,predicted_index] = -11100000
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    print(predicted_token)