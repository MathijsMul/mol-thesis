"""
analyze data distribution
"""

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_file(data_file):
    rels = ['#', 'v', '<', '>', '|', '=', '^']

    freq = []
    freq_idx = []
    freq_dict = Counter()

    for rel in rels:
        freq_dict[rel] = 0

    total = 0

    with open(data_file, 'r') as f:
        for idx, line in enumerate(f):
            all = line.split('\t')
            label = all[0]
            freq_dict[label] += 1
            freq_idx += [rels.index(label)]
            freq += [label]
            total += 1

    rel_freq_dict = Counter()

    for rel in rels:
        rel_freq = round(100 * freq_dict[rel] / total, 2)
        rel_freq_dict[rel] = str(rel_freq)

    def dic_to_tex(dic):
        rels_to_tex = {'#': '\#', '<': '<', '=': '=', '>': '>', '^': '\wedge', 'v': '\lor', '|': '\mid'}
        s = '$\{'
        for key in dic.keys():
            s += rels_to_tex[key] + ' : ' + dic[key] + '\%, '
        s += '\}$'
        return(s)

    return(total, freq_dict, rel_freq_dict, freq_idx)

if False:
    f1 = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_train.txt'
    dist_f1 = analyze_file(f1)
    folf1 = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_train_translated_from_nl.txt'
    dist_folf1 = analyze_file(folf1)

    unary_balanced_fol = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_data1_animals_train.txt_inddown_0.7'
    dist_unary_balanced_fol = analyze_file(unary_balanced_fol)

def visualize_label_dist(dist1,dist2):
    rels = ['#', 'v', '<', '>', '|', '=', '^']
    plt.hist([dist1,dist2], color=['#deebf7','#3182bd'], bins=np.arange(8),label=['train', 'test'], normed=1) #middle color:  '#9ecae1'
    plt.xticks(np.arange(8)+0.5,rels)
    plt.legend()
    plt.savefig('binaryfol_traintest_histograms')

def visualize_one_label_dist(dist):
    rels = ['#', 'v', '<', '>', '|', '=', '^']
    plt.hist(dist, color='#3182bd', rwidth=0.4, bins=np.arange(8))
    plt.xticks(np.arange(8)+0.5,rels)
    plt.legend()
    plt.savefig('binaryfol_dist')