import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from diagnostic_hypothesis import diagnostic_labels
from diagnostic_regression import load_model
from rnn import RNN
from sklearn import metrics


def predicted_labels(sentence, hypothesis, classifier, network='best-GRU'):
    if network == 'best-GRU':
        # load model
        vocab = ['Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not',
                 'some']
        rels = ['#', '<', '=', '>', '^', 'v', '|']

        word_dim = 25
        n_hidden = 128
        cpr_dim = 75

        model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/models/GRUbinary_2dets_4negs_train_0bracket_pairs1.pt'
        net = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
        net.load_state_dict(torch.load(model_path))

    s = [sentence.split()]

    _, hidden_vectors = net.rnn_forward(s, 1, hypothesis=hypothesis)
    test_hiddens = np.array(hidden_vectors[0])
    y_pred = classifier.predict(test_hiddens)

    labels = np.array([y_pred])

    return(labels)

def compare_and_plot(sentence, hypothesis, classifier, heatmaps, show_scores=False):
    """
    compare mae score for predictions and plot heatmaps together

    :param sentence:
    :return:
    """

    s = sentence.split()

    length = len(s)
    labels = [np.array(diagnostic_labels([s], hypothesis)), predicted_labels(sentence, hypothesis, classifier)]

    if heatmaps:
        rows = 2
        cols = 1
        fig, axn = plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(11,5))
        cbar_ax = fig.add_axes([.91, 0.05, .03, 0.9])

        text = np.array(s).reshape(1, length)

        color_palette = sns.color_palette('Blues', n_colors=3, desat=.8) # n_colors=9 for sentence idx
        print(labels)

        for i, ax in enumerate(axn.flat):
            show_cbar = (i == 0)
            sns.heatmap(labels[i], ax=ax, annot=text, annot_kws={"size": 15}, fmt='s', cmap=color_palette, xticklabels=False,yticklabels=False, cbar=show_cbar, cbar_ax=(None if i else cbar_ax))

        fig.tight_layout(rect=[0, 0, .9, 1])
        if show_scores:
            mae = round(metrics.mean_absolute_error(labels[0], labels[1]),2)
            mse = round(metrics.mean_squared_error(labels[0], labels[1]),2)
            plt.figtext(0,1,'MAE: ' + str(mae) + '\n MSE: ' + str(mse))

        s_summary = ''.join([word for word in s if word not in ['(', ')']])
        title = hypothesis + '_' + s_summary
        plt.savefig(title)
        plt.show()
        plt.clf()

# used for mon-dir illustrations
sentences = ['all children not like all not Europeans',
                 'some not Romans love some not children',
                 'not some Germans hate some not Romans']

for s in sentences:
    classifier = load_model('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/diag_class/gru/mon-dir/logreg_mon-dir.pkl')
    hypothesis = 'monotonicity_direction'
    compare_and_plot(s, hypothesis, classifier, heatmaps=False)
