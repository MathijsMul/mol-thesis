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

model_name = 'gru'
plotname = 'sentencevectors-gru-binaryfol-negations2.eps'
rels = ['#', '<', '=', '>', '^', 'v', '|']
word_dim = 25
n_hidden = 128
cpr_dim = 75

downsample_ratio = 0.025

if True:
#if model_name == 'trntn':
    vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
#elif model_name == 'gru':
    #vocab = ['(', ')', 'Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']

projection_method = 'pca'

#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_negations.txt'
#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/generate_data/binary_2dets_4negs_test_negations_0bracket_pairs.txt'
#test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/generate_data/binary_2dets_4negs_test_negations.txt'

if model_name == 'trntn':
    test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_negations.txt'
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=False)
elif model_name == 'gru':
    test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_negations_0bracket_pairs.txt'
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=True)

if model_name == 'trntn':
    #model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/trntn/tRNTNbinary_2dets_4negs_train.pt'
    model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol/models/tRNTNbinary_2dets_4negs_train4.pt'
    model = tRNTN(vocab, rels, word_dim, cpr_dim, None, None)
    model.load_state_dict(torch.load(model_path))
elif model_name == 'gru':
    #model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/GRUbinary_2dets_4negs_train_ada_nodrop_3.pt'
    model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
    model = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
    model.load_state_dict(torch.load(model_path))

total_correct = 0
total = 0

sentence_to_idx = defaultdict(lambda: len(sentence_to_idx))
idx_to_sentence = dict()
idx_to_vector = dict()
idx_to_det = dict()
dets = ['all', 'some', 'not all', 'not some']
det_to_vec = {det : [] for det in dets}

relations = []

rel_count = 0
for idx, item in enumerate(test_data.tree_data):
    if random.random() < downsample_ratio:
        instance = dat.SentencePairsDataset(None)
        instance.tree_data = [item]

        correct = compute_accuracy(instance, rels, model, print_outputs=False, confusion_matrix=False, batch_size=1)

        if correct == '100.00':
            if model_name == 'trntn':
                # for tree
                left_sentence = ' '.join(item[1].leaves())
                right_sentence = ' '.join(item[2].leaves())
            elif model_name == 'gru':
                # for recurrent
                left_sentence = ' '.join(item[1])
                right_sentence = ' '.join(item[2])

            idx_to_sentence[sentence_to_idx[left_sentence]] = left_sentence
            idx_to_sentence[sentence_to_idx[right_sentence]] = right_sentence

            if model_name == 'trntn':
                # for tree:
                idx_to_vector[sentence_to_idx[left_sentence]] = model.get_sentence_vector(item[1])
                idx_to_vector[sentence_to_idx[right_sentence]] = model.get_sentence_vector(item[2])
            elif model_name == 'gru':
                # for recurrent:
                idx_to_vector[sentence_to_idx[left_sentence]] = model.get_sentence_vector([item[1]])[0]
                idx_to_vector[sentence_to_idx[right_sentence]] = model.get_sentence_vector([item[2]])[0]

            left_idx = sentence_to_idx[left_sentence]
            right_idx = sentence_to_idx[right_sentence]

            relations += [[left_idx, right_idx]]

            if model_name == 'gru':
                # for recurrent
                #print(item)
                if item[1][0] in ['all', 'some']:
                    det_to_vec[item[1][0]] += [left_idx]
                    idx_to_det[left_idx] = item[1][0]
                elif item[1][0] == 'not':
                    if item[1][1] == 'some':
                        det_to_vec['not some'] += [left_idx]
                        idx_to_det[left_idx] = 'not some'
                    elif item[1][1] == 'all':
                        det_to_vec['not all'] += [left_idx]
                        idx_to_det[left_idx] = 'not all'

                if item[2][0] in ['all', 'some']:
                    det_to_vec[item[2][0]] += [right_idx]
                    idx_to_det[right_idx] = item[2][0]
                elif item[2][0] == 'not':
                    if item[2][1] == 'some':
                        det_to_vec['not some'] += [right_idx]
                        idx_to_det[right_idx] = 'not some'
                    elif item[2][1] == 'all':
                        det_to_vec['not all'] += [right_idx]
                        idx_to_det[right_idx] = 'not all'
            # else:
            #     print('Misclassified')

            if model_name == 'trntn':
                # for tree-shaped
                if item[1].leaves()[0] in ['all', 'some']:
                    det_to_vec[item[1].leaves()[0]] += [left_idx]
                    idx_to_det[left_idx] = item[1].leaves()[0]
                elif item[1].leaves()[0] == 'not':
                    if item[1].leaves()[1] == 'some':
                        det_to_vec['not some'] += [left_idx]
                        idx_to_det[left_idx] = 'not some'
                    elif item[1].leaves()[1] == 'all':
                        det_to_vec['not all'] += [left_idx]
                        idx_to_det[left_idx] = 'not all'

                if item[2].leaves()[0] in ['all', 'some']:
                    det_to_vec[item[2].leaves()[0]] += [right_idx]
                    idx_to_det[right_idx] = item[2].leaves()[0]
                elif item[2].leaves()[0] == 'not':
                    if item[2].leaves()[1] == 'some':
                        det_to_vec['not some'] += [right_idx]
                        idx_to_det[right_idx] = 'not some'
                    elif item[2].leaves()[1] == 'all':
                        det_to_vec['not all'] += [right_idx]
                        idx_to_det[right_idx] = 'not all'
        else:
            print('Misclassified')

print(det_to_vec)

#TODO: CONNECT NEGATED SENTENCE VECTORS

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

#print(vectors_to_plot)
if projection_method == 'pca':
    idx_to_projection = pca(vectors_to_plot, 2)
elif projection_method == 'tsne':
    idx_to_projection = tsne(vectors_to_plot, 2)

colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3']
red5 = '#67001f'
red4 = '#b2182b'
red3 = '#d6604d'
red2 = '#f4a582'
red1 = '#fddbc7'
neutral = '#f7f7f7'
blue1 = '#d1e5f0'
blue2 = '#92c5de'
blue3 = '#4393c3'
blue4 = '#2166ac'
blue5 = '#053061'

#det_to_color = {dets[i]: colors[i] for i in range(len(dets))}
det_to_color = {'all':red2, 'not some':red3,'some': blue2, 'not all':blue3}

print('Nr. of points: ', str(len(idx_to_projection)))

for point in idx_to_projection.items():
    idx = point[0]

    plt.scatter(point[1][0], point[1][1], s= 10, color=det_to_color[idx_to_det[idx]])

    #plt.annotate(idx, (point[1][0], point[1][1]), size=8)
    # to annotate:
    # plt.annotate(idx_to_sentence[idx], (point[1][0], point[1][1]), size=4)

for idx, pair in enumerate(relations):
    x1, y1 = idx_to_projection[pair[0]]
    x2, y2 = idx_to_projection[pair[1]]
    line = plt.plot([x1, x2], [y1, y2], color='black', lw=0.04,alpha=0.5)

print(idx_to_sentence)

ymin, ymax = plt.gca().get_ylim()
xmin, xmax = plt.gca().get_xlim()

legend_x, legend_y = 2 * xmax, 2 * ymax
line_1 = plt.scatter(legend_x, legend_y, label='Line 1', color=red2,s=10)
line_2 = plt.scatter(legend_x, legend_y, label='Line 2', color=red3,s=10)
line_3 = plt.scatter(legend_x, legend_y, label='Line 3', color=blue2,s=10)
line_4 = plt.scatter(legend_x, legend_y, label='Line 4', color=blue3,s=10)
plt.legend([line_1, line_2, line_3, line_4], ['all', 'not some', 'some', 'not all'])# numpoints=1)

plt.xlim(1.05*xmin, 1.05*xmax)
plt.ylim(1.05*ymin, 1.05*ymax)

plt.savefig(plotname, format='eps', dpi=500)
plt.show()
