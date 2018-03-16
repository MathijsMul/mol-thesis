import datamanager as dat
from test import compute_accuracy
from trntn import tRNTN
from trnn import tRNN
from rnn import RNN
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

word_dim = 25
cpr_dim = 75
n_hidden = 128
bound_layers = None
bound_embeddings = None

vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']
# model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol/models/tRNTNbinary_2dets_4negs_train4.pt'
# model = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)

#model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs5.pt'
model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/LSTMbinary_2dets_4negs_train_0bracket_pairs4.pt'
model= RNN('LSTM', vocab, rels, word_dim, n_hidden, cpr_dim)
model.load_state_dict(torch.load(model_path))

# unary_vocab = ['all', 'growl', 'lt_three', 'lt_two', 'mammals', 'most', 'move', 'no', 'not', 'not_all', 'not_most', 'pets', 'reptiles', 'some', 'swim', 'three', 'turtles', 'two', 'walk', 'warthogs']
# unary_rels = ['#', '<', '=', '>', '^', 'v', '|']
# best_unary_model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/bowman_rep/models/tRNTNf3_train1.pt'
# unary_model = tRNTN(unary_vocab, unary_rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
# unary_model.load_state_dict(torch.load(best_unary_model_path))

def show_pca(n, model, vocab):

    pca = PCA(n_components=n)
    word_embeddings = model.voc.weight.data.numpy()
    pca.fit(word_embeddings)

    projections = pca.transform(word_embeddings)
    x = projections[:,0]
    y = projections[:,1]

    if n == 2:
        plt.scatter(x, y,s=10)

        for i, txt in enumerate(vocab):
            plt.annotate(txt, (x[i], y[i]),fontsize=10)

    elif n == 3:
        fig = plt.figure(1, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        z = projections[:, 2]

        # Reorder the labels to have colors matching the cluster results
        #y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(x,y,z, c=y, cmap=plt.cm.spectral,
                   edgecolor='k')

        for i, txt in enumerate(vocab):
            ax.text(x[i], y[i], z[i], '%s' % (txt), size=5, zorder=1,color='k')

    #plt.title('PCA for learned word embeddings')
    #plt.savefig('pca_words_besttrntn_binaryfol')
    plt.savefig('pca_words_bestlstm_binaryfol.eps', format='eps', dpi=500)
    plt.show()

def make_bubbleplot(model_paths, vocab, rels):
    nr_models = len(model_paths)
    nr_words = len(vocab)
    x_values = np.zeros((nr_models, nr_words))
    y_values = np.zeros((nr_models, nr_words))

    for idx, path in enumerate(model_paths):
        # model = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim, bound_layers=bound_layers,
        #               bound_embeddings=bound_embeddings)

        model = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
        model.load_state_dict(torch.load(path))

        pca = PCA(n_components=2)
        word_embeddings = model.voc.weight.data.numpy()
        pca.fit(word_embeddings)

        projections = pca.transform(word_embeddings)
        x = projections[:, 0]
        y = projections[:, 1]

        # x_values = np.vstack([x_values, x])
        # y_values = np.vstack([y_values, y])
        x_values[idx,:] = x
        y_values[idx,:] = y

    x_mean = np.mean(x_values, axis=0)
    y_mean = np.mean(y_values, axis=0)

    distances = np.sqrt((x_values-x_mean)**2 + (y_values-y_mean)**2)
    mean_distances = np.mean(distances,axis=0)
    print(mean_distances)

    fig, ax = plt.subplots()

    # 25000 is the correct point-distance ratio
    plt.scatter(x_mean, y_mean, s=25000*mean_distances, c=mean_distances, cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=1)
    plt.scatter(x_mean, y_mean, s=10)

    for i, txt in enumerate(vocab):
        #circle = plt.Circle((x_mean[i], y_mean[i]), mean_distances[i], color='g', clip_on=False)
        #ax.add_artist(circle)
        if i != 8:
            plt.annotate(txt, (x_mean[i], y_mean[i]),fontsize=10)
        # correct like
        else:
            plt.annotate(txt, (x_mean[i]-0.1, y_mean[i]), fontsize=10)

    plt.savefig('avg_word_emb_dist_grus_binaryfol',dpi=1000)
    #plt.savefig('avg_word_emb_dist_trntns_binaryfol.eps', format='eps', dpi=1000)
    #plt.show()

#show_pca(2, unary_model, unary_vocab)
show_pca(2, model, vocab)
exit()

model_paths=['/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt',
             '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs2.pt',
             '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs3.pt',
             '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs4.pt',
             '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs5.pt']

make_bubbleplot(model_paths, vocab, rels)
