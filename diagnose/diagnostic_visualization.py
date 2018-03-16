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

def compare_and_plot(sentence, hypothesis, classifier, heatmaps, show_scores=False, latex_table=True):
    """
    compare mae score for predictions and plot heatmaps together

    :param sentence:
    :return:
    """

    s = sentence.split()

    length = len(s)
    labels = [np.array(diagnostic_labels([s], hypothesis)), predicted_labels(sentence, hypothesis, classifier)]


    if latex_table:

        print('sentence:')
        print(s)

        print('hypothesis:')
        hyps = labels[0] - 1
        print(hyps)
        print('prediction:')
        preds = labels[1] - 1
        print(preds)

        hyp_list = hyps.tolist()[0]

        hyp_row = ''
        pred_row = ''


        #print(hyp_list)

        for idx, hyp in enumerate(hyp_list):
            if hyp == 0:
                hyp_row += '-'
            elif hyp == '-1':
                hyp_row += 'downarrow$'
            elif hyp == '1':
                hyp_row += 'uparrow$'
            #if idx <
            hyp_row += '&'
        hyp_row += '\\'

        #print(hyp_row)

    if heatmaps:
        rows = 2
        cols = 1
        fig, axn = plt.subplots(rows,cols,sharex=True,sharey=True,figsize=(11,5))
        cbar_ax = fig.add_axes([.91, 0.05, .03, 0.9])


        text = np.array(s).reshape(1, length)

        #color_palette = sns.color_palette('Blues', n_colors=9, desat=.8)
        color_palette = sns.color_palette('Blues', n_colors=3, desat=.8)
        ticks = np.arange(length)
        print(labels)

        for i, ax in enumerate(axn.flat):
            # if hypothesis == 'pos':
            #     pos_labels = ['bracket', 'noun', 'verb', 'quant', 'neg']
            #     ticks = np.array(pos_labels).reshape(1, 5)
            #sns.heatmap(labels[i], ax=ax, cbar=i==0, annot=text, fmt='s', cmap=color_palette, xticklabels=False, yticklabels=False, cbar_ax=None if i else cbar_ax, cbar_kws={"ticks": ticks})
            show_cbar = (i == 0)
            sns.heatmap(labels[i], ax=ax, annot=text, annot_kws={"size": 15}, fmt='s', cmap=color_palette, xticklabels=False,yticklabels=False, cbar=show_cbar, cbar_ax=(None if i else cbar_ax))

            # cbar = fig.colorbar(hm, ticks=[-1, 0, 1])
            # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
            # colorbar = hm.collections[0].colorbar
            # colorbar.set_ticks([-0.5,0,0.5])
            #cbar_ax.set_ticklabels(['1', '2', '3'])

        fig.tight_layout(rect=[0, 0, .9, 1])
        if show_scores:
            mae = round(metrics.mean_absolute_error(labels[0], labels[1]),2)
            mse = round(metrics.mean_squared_error(labels[0], labels[1]),2)
            plt.figtext(0,1,'MAE: ' + str(mae) + '\n MSE: ' + str(mse))

        #plt.title(hypothesis)
        s_summary = ''.join([word for word in s if word not in ['(', ')']])
        title = hypothesis + '_' + s_summary
        plt.savefig(title)
        plt.show()
        plt.clf()


# sentences = ['( ( some Italians  ) ( ( not love ) ( some Romans ) ) )','( ( all Europeans  ) ( ( not love ) ( all ( not Italians ) ) ) )',
#              '( ( all ( not Germans )  ) ( love ( some Europeans ) ) )', '( ( ( not all ) ( not Europeans )  ) ( love ( some Europeans ) ) )',
#             '( ( some Europeans  ) ( ( not hate ) ( all Europeans ) ) )', '( ( ( not all ) Romans  ) ( love ( some Romans ) ) ) ']

# sentences = ['all Europeans not like all not Italians',
#              'not some Germans hate some Romans',
#              'not all not children fear all Europeans']
    # 'some not Italians not love some Romans']#,

# used for mon-dir illustrations
sentences = ['all children not like all not Europeans',
                 'some not Romans love some not children',
                 'not some Germans hate some not Romans']


for s in sentences:
    classifier = load_model('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/diag_class/gru/mon-dir/logreg_mon-dir.pkl')
    hypothesis = 'monotonicity_direction'
    compare_and_plot(s, hypothesis, classifier, heatmaps=False)

# TODO:
# change color palette for pos tags, because this is not a continuous scale
# map number categories to POS tag strings for pos plots