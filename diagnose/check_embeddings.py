
import datamanager as dat
from test import compute_accuracy
from srn import SRN
from trntn import tRNTN
from trnn import tRNN
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
#vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some', 'two']
rels = ['#', '<', '=', '>', '^', 'v', '|']
word_dim = 25
cpr_dim = 75
bound_layers = None
bound_embeddings = None

path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/trntn/tRNTNbinary_2dets_4negs_train.pt'
#path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/trntn/tRNTNbinary3dets_4negs_train.pt'
model = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
model.load_state_dict(torch.load(path))

def show_pca(n, model):

    pca = PCA(n_components=3)
    word_embeddings = model.voc.weight.data.numpy()
    pca.fit(word_embeddings)

    projections = pca.transform(word_embeddings)
    x = projections[:,0]
    y = projections[:,1]

    if n == 2:
        plt.scatter(x, y)

        for i, txt in enumerate(vocab):
            plt.annotate(txt, (x[i], y[i]))

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

    plt.title('PCA for learned word embeddings')
    plt.show()

show_pca(3, model)