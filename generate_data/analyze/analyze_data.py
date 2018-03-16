from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_file(data_file):
    # rels = ['=', '<', '>', 'v', '^', '|', '#']
    # rels = [0,1,2,3,4]
    # rels = ['0', '1', '2', '3', '4']
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

    #{'=': 0.001, '$<$': 0.02, '$>$': 0.02, '$\lor$': 0.02, '$\wedge$': 0.001, '|': 0.02, '\#': 0.9}

    print(dic_to_tex(rel_freq_dict))

    #norm_freq_idx = []

    # return(total, freq_dict, rel_freq_dict)

    #return(rel_freq_dict)
    return(freq_idx)
    #return(freq)

if False:
    f1 = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/bowman/f1_train.txt'
    dist_f1 = analyze_file(f1)
    folf1 = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_train_translated_from_nl.txt'
    dist_folf1 = analyze_file(folf1)

    unary_balanced_fol = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_data1_animals_train.txt_inddown_0.7'
    dist_unary_balanced_fol = analyze_file(unary_balanced_fol)

def visualize_label_dist(dist1,dist2):
    rels = ['#', 'v', '<', '>', '|', '=', '^']
    #middle color:  '#9ecae1'
    plt.hist([dist1,dist2], color=['#deebf7','#3182bd'], bins=np.arange(8),label=['train', 'test'], normed=1)
    plt.xticks(np.arange(8)+0.5,rels)
    plt.legend()
    plt.savefig('binaryfol_traintest_histograms')

def visualize_one_label_dist(dist):
    rels = ['#', 'v', '<', '>', '|', '=', '^']
    plt.hist(dist, color='#3182bd', rwidth=0.4, bins=np.arange(8))
    plt.xticks(np.arange(8)+0.5,rels)
    plt.legend()
    plt.savefig('binaryfol_dist')

binarytrain = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
binarytraindist = analyze_file(binarytrain)
binarytest = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
binarytestdist = analyze_file(binarytest)

visualize_label_dist(binarytraindist, binarytestdist)
