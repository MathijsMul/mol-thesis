from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import random

def compare_files(nl_file, fol_file,plot=False):
    rels = ['#', '<', '=', '>', '^', 'v', '|']

    #conflicts_counter = Counter()
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

                                #if rel_nl != rel_fol:
                                    #conflicts_counter[rel_nl + rel_fol] += 1

                                conf_matrix[rels.index(rel_fol)][rels.index(rel_nl)] += 1
                                df = df.append({'FOL': float(rels.index(rel_fol)) + random.gauss(0,0.01), 'NL':float(rels.index(rel_nl)) + random.gauss(0,0.01)}, ignore_index=True)

            if idxfol == 500:
                break

        #conf_matrix = conf_matrix.numpy()
        #print('Conflicts:')
        #print(conf_matrix)
        #print('Total nr of conflicts')
        #print(conf_matrix.sum())
        if plot:

            print(conf_matrix)
            print(conf_matrix.numpy().sum())

            # normalize
            #for i in range(7):
            #   conf_matrix[i] = conf_matrix[i] / conf_matrix[i].sum()

            confusion = pd.DataFrame(conf_matrix.numpy(), index=rels, columns=rels)
            h = sns.heatmap(confusion, cmap='Blues')
            h.set_yticklabels(rels, rotation=0)

            plot_name = 'nonnorm_conf_compare_nl_fol_trainf1'

            plt.savefig(plot_name)

            plt.close()
        return(df)

    #return(conflicts_counter)

#
# data1 = np.random.multivariate_normal([0,0], [[1,0.5],[0.5,1]], size=200)
# data2 = np.random.multivariate_normal([0,0], [[1,-0.8],[-0.8,1]], size=100)
#
# # both df1 and df2 have bivaraite normals, df1.size=200, df2.size=100
# df1 = pd.DataFrame(data1, columns=['x1', 'y1'])
# df2 = pd.DataFrame(data2, columns=['x2', 'y2'])
# print(df1)
# print(df1.x1)
# print(df1.y1)
#
# # plot
# # ========================================
# graph = sns.jointplot(x=df1.x1, y=df1.y1, color='r')
# plt.show()
# exit()

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
    :return:
    """

    #cumulative_counts = sum(counters, Counter())

    # average wrt number of samples
    #for key in cumulative_counts:
    #    cumulative_counts[key] /= SAMPLES

    # plot

    #labels, values = zip(*cumulative_counts.items())
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
    #plt.show()
    plt.close()

# c = Counter({'v#': 2356, '|#': 2020, '>#': 1425, '<#': 1389, '<>': 638, '><': 632, '|v': 517, 'v|': 401, '|<': 136, 'v>': 136, '|^': 75, 'v^': 75, '#<': 24, '#>': 24, '#|': 22, '#v': 22})
# print(sum(c.values()))

#print(989200/21856)

total = 0
ind_count = 0
with open(nl_train, 'r') as nl_f:
    for idx, line in enumerate(nl_f):
        all = line.split('\t')
        #print(all)
        rel_fol = all[0]
        if rel_fol == '#':
            ind_count += 1
        total += 1
print('total:')
print(total)
print('independent:')
print(ind_count)

print(100*(ind_count/total))

