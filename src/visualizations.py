import argparse
import logging

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np


# code for PCA visualizations and bert emmbeddings extraction


def draw_interactive(x, y, text):
    """
    Draw a plot visualizing word vectors with the posibility to hover over a datapoint and see
    a word associating with it

    :param      x:     A list of values for the x-axis
    :type       x:     list
    :param      y:     A list of values for the y-axis
    :type       y:     list
    :param      text:  A list of textual values associated with each (x, y) datapoint
    :type       text:  list
    """
    norm = plt.Normalize(1, 4)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots()
    sc = plt.scatter(x, y, c='b', s=100, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        note = "{}".format(" ".join([text[n] for n in ind["ind"]]))
        annot.set_text(note)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def load(fname):
    try:
        with open(fname, 'r') as f:
            V, H = (int(a) for a in next(f).split())
            W, i2w, w2i = np.zeros((V, H)), [], {}
            for i, line in enumerate(f):
                parts = line.split()
                word = parts[0].strip()
                w2i[word] = i
                W[i] = list(map(float, parts[1:]))
                i2w.append(word)
            return W, i2w, w2i, V, H
    except:
        print("Error: failing to load the model to the file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding visualization toolkit')
    parser.add_argument('file', type=str, help='A textual file containing word vectors')
    parser.add_argument('-v', '--vector-type', default='w2v', choices=['w2v', 'ri'])
    parser.add_argument('-d', '--decomposition', default='svd', choices=['svd', 'pca'],
                        help='Your favorite decomposition method')
    args = parser.parse_args()

    W, i2w, w2i, V, H = load(args.file)
    x = []
    y = []
    text = list(w2i.keys())

    if args.decomposition == 'svd':
        svd = TruncatedSVD(n_components=2)
        svd.fit_transform(W)
        x = svd.components_[0]
        y = svd.components_[1]
    elif args.decomposition == 'pca':
        pca = PCA(n_components=2)
        pca.fit_transform(W)
        x = pca.components_[0]
        y = pca.components_[1]

    draw_interactive(x, y, text)

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
    segments_ids[0] = 0
    segments_ids[1] = 0

    # Convert inputs to torch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])


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
        predictions[0, masked_index, predicted_index] = -11100000
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        print(predicted_token)

