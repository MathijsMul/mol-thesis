"""
plot sentence vectors as outputted by different models,
connect vectors according to their true entailment relation
(only for correct classifications)
"""

import torch
from rnn import RNN
from trntn import tRNTN
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

# random.seed(9001)

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
# n_hidden = 128
cpr_dim = 75

test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=False)

model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol/models/tRNTNbinary_2dets_4negs_train4.pt'
trntn = tRNTN(vocab, rels, word_dim, cpr_dim, None, None)
trntn.load_state_dict(torch.load(model_path))

total_correct = 0
total = 0

downsample_ratio = 0.03
sentence_to_idx = defaultdict(lambda: len(sentence_to_idx))
idx_to_sentence = dict()
idx_to_vector = dict()
# idx_to_pca_vector = dict()
relation_lines = {}
for rel in rels:
    relation_lines[rel] = []

rel_count = 0
for idx, item in enumerate(test_data.tree_data):
    if random.random() < downsample_ratio:
        instance = dat.SentencePairsDataset(None)
        instance.tree_data = [item]
        correct = compute_accuracy(instance, rels, trntn, print_outputs=False, confusion_matrix=False, batch_size=1)

        if correct == '100.00':
            relation = item[0]
            left_sentence = ' '.join(item[1].leaves())
            right_sentence = ' '.join(item[2].leaves())
            idx_to_sentence[sentence_to_idx[left_sentence]] = left_sentence
            idx_to_sentence[sentence_to_idx[right_sentence]] = right_sentence
            relation_lines[relation] += [[sentence_to_idx[left_sentence], sentence_to_idx[right_sentence]]]
            idx_to_vector[sentence_to_idx[left_sentence]] = trntn.get_sentence_vector(item[1])
            idx_to_vector[sentence_to_idx[right_sentence]] = trntn.get_sentence_vector(item[2])
            rel_count += 1

print('Count: ', str(rel_count))
vectors_to_plot = torch.stack(idx_to_vector.values()).data.numpy()

def pca(vectors, out_dim):
    pca = PCA(n_components=out_dim)
    pca.fit(vectors)
    projections = pca.transform(vectors)
    idx_to_pca_vector = {idx: projections[idx, :] for idx in range(vectors.shape[0])}
    return (idx_to_pca_vector)

def tsne(vectors, out_dim):
    tsne = TSNE(n_components=out_dim)
    projections = tsne.fit_transform(vectors)
    idx_to_tsne_vector = {idx: projections[idx, :] for idx in range(vectors.shape[0])}
    return (idx_to_tsne_vector)

idx_to_projection = pca(vectors_to_plot, 2)

for point in idx_to_projection.items():
    plt.scatter(point[1][0], point[1][1], color='black', s=2)
    idx = point[0]
    # to annotate points:
    # plt.annotate(idx, (point[1][0], point[1][1]), size=8)

colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f']
rel_to_color = {rels[i]: colors[i] for i in range(len(rels))}
two_colors = ['#4393c3','#d6604d']

lines = []
handles = []
for rel in rels:
    for idx, pair in enumerate(relation_lines[rel]):
        x1, y1 = idx_to_projection[pair[0]]
        x2, y2 = idx_to_projection[pair[1]]
        line = plt.plot([x1, x2], [y1, y2], color=rel_to_color[rel], lw=0.5)

line_1, = plt.plot([0, 0], [0, 0], label='Line 1', color=colors[0])
line_2, = plt.plot([0, 0], [0, 0], label='Line 2', color=colors[1])
line_3, = plt.plot([0, 0], [0, 0], label='Line 3', color=colors[2])
line_4, = plt.plot([0, 0], [0, 0], label='Line 4', color=colors[3])
line_5, = plt.plot([0, 0], [0, 0], label='Line 5', color=colors[4])
line_6, = plt.plot([0, 0], [0, 0], label='Line 6', color=colors[5])
line_7, = plt.plot([0, 0], [0, 0], label='Line 7', color=colors[6])
plt.legend([line_1, line_2, line_3, line_4, line_5, line_6, line_7], rels)

plt.savefig('sentencevectors-trntn-binaryfol-relations8.png', format='png', dpi=500)
plt.show()
