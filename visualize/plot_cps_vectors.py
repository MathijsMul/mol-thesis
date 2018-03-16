"""
plot PCA-projections of all representations preceding sentence vector,
i.e. by successively applying the learned composition function to word embeddings
and intermediate compositions
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

#random.seed(9001)

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
#n_hidden = 128
cpr_dim = 75

test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=False)

model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol/models/tRNTNbinary_2dets_4negs_train4.pt'
trntn = tRNTN(vocab, rels, word_dim, cpr_dim, None, None)
trntn.load_state_dict(torch.load(model_path))

downsample_ratio = 1
dim_red = 'PCA' # or 'tSNE'
annotate = False
connect = False

def plot_cps_vectors(save, nr_points_to_plot):
    nr_points_plotted = 0

    for idx, item in enumerate(test_data.tree_data):
        if random.random() < downsample_ratio:
            instance = dat.SentencePairsDataset(None)
            instance.tree_data = [item]
            correct = compute_accuracy(instance, rels, trntn, print_outputs=False, confusion_matrix=False, batch_size=1)

            if correct == '100.00':
                plotted_sentence = ''.join(item[1].leaves())

                # only left sentence considered
                # print(idx)
                vector_to_idx, idx_to_vector, connections, idx_to_word = trntn.get_cps_vectors(item[1])
                #print(vector_to_idx, idx_to_vector, connections, idx_to_word)
                nr_points_plotted += 1

                #break

                vectors_to_plot = torch.stack(idx_to_vector.values()).numpy()

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

                num_points = len(idx_to_projection)
                for i, point in enumerate(idx_to_projection.items()):
                    if i == (num_points - 1):
                        # plot sentence vector in different color
                        plt.scatter(point[1][0], point[1][1], color='#d6604d', s=3)
                        #plt.annotate(plotted_sentence, (point[1][0], point[1][1]), size=8, color='black')
                    else:
                        plt.scatter(point[1][0], point[1][1], color='#4393c3', s=3)

                    if annotate:
                        idx = point[0]
                        plt.annotate(idx_to_word[idx], (point[1][0], point[1][1]), size=8, color='#1f78b4')

                if connect:
                    max_length = 0
                    for pair in connections.values():
                        x1, y1 = idx_to_projection[pair[0]]
                        x2, y2 = idx_to_projection[pair[1]]
                        full_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if full_length > max_length:
                            max_length = full_length

                    for pair in connections.values():
                        # print(pair)
                        x1, y1 = idx_to_projection[pair[0]]
                        x2, y2 = idx_to_projection[pair[1]]
                        full_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        dx = (0.89 + 0.08 * (full_length / max_length)) * (x2 - x1)
                        dy = (0.89 + 0.08 * (full_length / max_length)) * (y2 - y1)
                        if dim_red == 'PCA':
                            line = plt.arrow(x1, y1, dx, dy, color='black', lw=0.3, head_width=0.02)
                        elif dim_red == 'tSNE':
                            line = plt.arrow(x1, y1, dx, dy, color='black', lw=0.3, head_width=5)

                if nr_points_plotted == nr_points_to_plot:
                    break


    plot_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/plots/cps_vectors/' + plotted_sentence #+ '.eps'

    if save:
        #plt.savefig(plot_path, format='eps', dpi=1000)
        plt.savefig(plot_path)
    plt.show()
    #plt.clf()


plot_cps_vectors(save=True,nr_points_to_plot=1000)