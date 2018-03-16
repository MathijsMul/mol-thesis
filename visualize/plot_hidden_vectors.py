"""
plot PCA-projections of all hidden units preceding sentence vector for RNNs
"""

import torch
from rnn import RNN
import datamanager as dat
from test import compute_accuracy
import random
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.manifold import TSNE
import numpy as np

#random.seed(9001)

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']
word_dim = 25
n_hidden = 128
cpr_dim = 75

#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=True)

#model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/GRUbinary_2dets_4negs_train_ada_nodrop_3.pt'
model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
gru = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
gru.load_state_dict(torch.load(model_path))

downsample_ratio = 0.005
dim_red = 'PCA' # or 'tSNE'
#sentence_to_idx = defaultdict(lambda: len(sentence_to_idx))
#idx_to_sentence = dict()
#idx_to_vector = dict()
#idx_to_pca_vector = dict()
#relation_lines = {}

def generate_plot():
    for idx, item in enumerate(test_data.tree_data):
        if random.random() < downsample_ratio:
            instance = dat.SentencePairsDataset(None)
            instance.tree_data = [item]
            correct = compute_accuracy(instance, rels, gru, print_outputs=False, confusion_matrix=False, batch_size=1)

            if correct == '100.00':
                sentence = ''.join(item[1])
                if item[1][0] == 'some':
                    # only left sentence considered
                    hidden_units = gru.get_hidden_vectors([item[1]])[0]
                    idx_to_vector = {i : np.array(hidden_units[i]) for i in range(len(hidden_units))}
                    idx_to_word = {i : item[1][i] for i in range(len(item[1]))}
                    connections = [[i, i + 1] for i in range(len(item[1]) - 1)]
        #            vector_to_idx, idx_to_vector, connections, idx_to_word = gru.get_hidden_vectors([item[1]])
        #            print(vector_to_idx, idx_to_vector, connections, idx_to_word)
                    break

    vectors_to_plot = np.stack(idx_to_vector.values())

    def pca(vectors, out_dim):
        pca = PCA(n_components=out_dim)
        pca.fit(vectors)
        projections = pca.transform(vectors)
        idx_to_pca_vector = {idx : projections[idx,:] for idx in range(vectors.shape[0])}
        return(idx_to_pca_vector)

    def tsne(vectors, out_dim):
        tsne = TSNE(n_components=out_dim)
        projections = tsne.fit_transform(vectors)
        idx_to_tsne_vector = {idx : projections[idx,:] for idx in range(vectors.shape[0])}
        return(idx_to_tsne_vector)

    if dim_red == 'PCA':
        idx_to_projection = pca(vectors_to_plot, 2)
    elif dim_red == 'tSNE':
        idx_to_projection = tsne(vectors_to_plot, 2)

    for point in idx_to_projection.items():
        plt.scatter(point[1][0], point[1][1], color='black', s=3)
        idx = point[0]
        plt.annotate(idx_to_word[idx], (point[1][0], point[1][1]), size=10, color='#1f78b4')

    for pair in connections:
        #print(pair)
        x1, y1 = idx_to_projection[pair[0]]
        x2, y2 = idx_to_projection[pair[1]]
        to_right = x2 > x1
        to_up = y2 > y1

        if to_right:
            dx = 0.95* (x2 - x1) #- 0.15
        else:
            dx = 0.95* (x2 - x1) #+ 0.15
        if to_up:
            dy = 0.95* (y2 - y1) #- 0.15
        else:
            dy = 0.95* (y2 - y1) #+ 0.15

        if dim_red == 'PCA':
            line = plt.arrow(x1, y1, dx, dy, color='black', lw=0.5, head_width=0.1)
        elif dim_red == 'tSNE':
            line = plt.arrow(x1, y1, dx, dy, color='black', lw=0.5, head_width=5)

    plt.savefig(sentence, dpi=500)
    plt.clf()

for i in range(10):
    generate_plot()