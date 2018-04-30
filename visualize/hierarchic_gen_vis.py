"""
visualize results of hierarchic generalization experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

DET_SUBJ_LIST = ['some', 'all', 'not some', 'not all']
dir = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/hierarchic_gen/'
cols = ['model', 'd1', 'd2', 'training accuracy', 'testing accuracy']

def visualize_results(model, data, visualize):
    runs = ['1', '2', '3', '4', '5']
    data_list = []

    for det1 in DET_SUBJ_LIST[::-1]:
        results = []
        for det2 in DET_SUBJ_LIST:
            cum_train = 0
            cum_test = 0
            for run in runs:
                if det1 == 'not some':
                    det1_corr = 'not_some'
                elif det1 == 'not all':
                    det1_corr = 'not_all'
                else:
                    det1_corr = det1
                if det2 == 'not some':
                    det2_corr = 'not_some'
                elif det2 == 'not all':
                    det2_corr = 'not_all'
                else:
                    det2_corr = det2
                filename = dir + model + '/from_' + data + '/' + model + '_' + data + det1_corr + det2_corr + '_' + run + '.txt'
                with open(filename, 'r') as f:
                    for idx, line in enumerate(f):
                        if idx == 29:
                            test_acc = float(line.split()[1])
                            cum_test += test_acc
                        elif idx == 31:
                            train_acc = float(line.split()[1])
                            cum_train += train_acc
            avg_train = cum_train / 5
            avg_test = cum_test / 5
            data_list.append([model.upper(), det1, det2, avg_train, avg_test])

    df = pd.DataFrame(data_list, columns=cols)

    pivoted_result = df.pivot(index='d1', columns='d2', values='testing accuracy')
    print('mean ' + model + data)
    print(pivoted_result.mean(0).mean(0))

    if visualize:
        reordered = pivoted_result.reindex(DET_SUBJ_LIST, axis=0).reindex(DET_SUBJ_LIST, axis=1)
        color_palette = sns.color_palette('Blues', 255, desat=.8)
        ax = sns.heatmap(reordered, annot=True, cmap=color_palette)
        #plt.show()
        plt.savefig('hier_gen_' + model + '_' + data, dpi=500)
        plt.clf()

for model in ['srn', 'gru']:
    for data in ['standard', 'bulk']:
        result = visualize_results(model, data,visualize=False)
