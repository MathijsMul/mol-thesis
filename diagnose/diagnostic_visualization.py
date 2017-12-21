import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import math
import pickle
from diagnostic_hypothesis import diagnostic_labels
from diagnostic_regression import load_model
from rnn import RNN
from sklearn import metrics


def predicted_labels(sentence, hypothesis, classifier, network='default_GRU'):
    if network == 'default_GRU':
        # load model
        vocab = ['(', ')', 'Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like',
                 'love', 'not', 'some']
        rels = ['#', '<', '=', '>', '^', 'v', '|']

        word_dim = 25
        n_hidden = 128
        cpr_dim = 75

        model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/GRUbinary_2dets_4negs_train_ada_nodrop_3.pt'
        net = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
        net.load_state_dict(torch.load(model_path))

    s = [sentence.split()]

    _, hidden_vectors = net.rnn_forward(s, 1, hypothesis=hypothesis)
    test_hiddens = np.array(hidden_vectors[0])
    y_pred = classifier.predict(test_hiddens)

    labels = np.array([y_pred])

    return(labels)

def compare_and_plot(sentence, hypothesis, classifier, show_scores=False):
    """
    compare mae score for predictions and plot heatmaps together

    :param sentence:
    :return:
    """

    s = sentence.split()
    length = len(s)
    labels = [np.array(diagnostic_labels([s], hypothesis)), predicted_labels(sentence, hypothesis, classifier)]

    rows = 2
    cols = 1
    fig, axn = plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(11,5))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    text = np.array(s).reshape(1, length)

    color_palette = sns.color_palette('Blues', 255, desat=.8)

    ticks = np.arange(length)

    for i, ax in enumerate(axn.flat):
        # if hypothesis == 'pos':
        #     pos_labels = ['bracket', 'noun', 'verb', 'quant', 'neg']
        #     ticks = np.array(pos_labels).reshape(1, 5)
        #sns.heatmap(labels[i], ax=ax, cbar=i==0, annot=text, fmt='s', cmap=color_palette, xticklabels=False, yticklabels=False, cbar_ax=None if i else cbar_ax, cbar_kws={"ticks": ticks})
        sns.heatmap(labels[i], ax=ax, annot=text, fmt='s', cmap=color_palette, xticklabels=False,yticklabels=False, cbar=False)

        # cbar = fig.colorbar(hm, ticks=[-1, 0, 1])
        # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

    #fig.tight_layout(rect=[0, 0, .9, 1])
    if show_scores:
        mae = round(metrics.mean_absolute_error(labels[0], labels[1]),2)
        mse = round(metrics.mean_squared_error(labels[0], labels[1]),2)
        plt.figtext(0,1,'MAE: ' + str(mae) + '\n MSE: ' + str(mse))

    #plt.title(hypothesis)
    s_summary = ''.join([word for word in s if word not in ['(', ')']])
    title = hypothesis + '_' + s_summary
    plt.savefig(title)
    plt.close()


sentences = ['( ( some Italians  ) ( ( not love ) ( some Romans ) ) )','( ( all Europeans  ) ( ( not love ) ( all ( not Italians ) ) ) )',
             '( ( all ( not Germans )  ) ( love ( some Europeans ) ) )', '( ( ( not all ) ( not Europeans )  ) ( love ( some Europeans ) ) )',
            '( ( some Europeans  ) ( ( not hate ) ( all Europeans ) ) )', '( ( ( not all ) Romans  ) ( love ( some Romans ) ) ) ']

for s in sentences:
    classifier = load_model('classifiers/logreg_pos.pkl')
    hypothesis = 'pos'
    compare_and_plot(s, hypothesis, classifier)

# TODO:
# change color palette for pos tags, because this is not a continuous scale
# map number categories to POS tag strings for pos plots