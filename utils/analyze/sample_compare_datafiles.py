"""
compare data files
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import random

def compare_files(nl_file, fol_file,plot=False):
    rels = ['#', '<', '=', '>', '^', 'v', '|']
    conf_matrix = torch.zeros(7,7)
    df = pd.DataFrame(columns=['FOL','NL'])

    with open(fol_file, 'r') as fol_f:
        for idxfol, line in enumerate(fol_f):
            if idxfol % 100 == 0:
                print('Sentence ', idxfol)
            all = line.split('\t')
            left_fol = all[1]
            right_fol = all[2]
            rel_fol = all[0]
            with open(nl_file, 'r') as nl_f:
                for idxnl, line in enumerate(nl_f):
                    # idx of nl is at least as high as idx of fol analogue (because of most)
                    if idxnl >= idxfol:
                        all = line.split('\t')
                        left_nl = all[1]
                        if left_fol == left_nl:
                            right_nl = all[2]
                            if right_fol == right_nl:
                                rel_nl = all[0]
                                conf_matrix[rels.index(rel_fol)][rels.index(rel_nl)] += 1
                                df = df.append({'FOL': float(rels.index(rel_fol)) + random.gauss(0,0.01), 'NL':float(rels.index(rel_nl)) + random.gauss(0,0.01)}, ignore_index=True)

        if plot:
            confusion = pd.DataFrame(conf_matrix.numpy(), index=rels, columns=rels)
            h = sns.heatmap(confusion, cmap='Blues')
            h.set_yticklabels(rels, rotation=0)
            plot_name = 'nonnorm_conf_compare_nl_fol_trainf1'
            plt.savefig(plot_name)
            plt.close()
        return(df)

nl_train = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_train.txt'
fol_train = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_train_translated_from_nl.txt'
df = compare_files(nl_train, fol_train)
print(df)
sns.jointplot(x=df.FOL, y=df.NL,color='#3182bd',kind='scatter')
plt.show()
exit()

def visualize(counter):
    """
    :param counters: list of Counter() objects
    :return: plot
    """

    labels, values = zip(*counter.items())
    sorted_values = sorted(values)[::-1]
    sorted_labels = [x for (y,x) in sorted(zip(values,labels))][::-1]
    indexes = np.arange(len(sorted_labels))
    width = 1

    plt.bar(indexes, sorted_values)
    plt.xticks(indexes + width * 0.5, sorted_labels)
    plt.title("Frequencies of NL vs FOL conflicts in train data (total 22k instances)")
    plt.xlabel("Conflict")
    plt.ylabel("Frequency")
    plt.savefig("conflicts_fol_translation_of_nl_f1")
    plt.close()
