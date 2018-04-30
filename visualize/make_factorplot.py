"""
visualize results of partial bracketing experiments in factorplot
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_bracket_data():
    cols=['model', '% brackets', 'train_acc', 'testing accuracy']
    data_list = []
    models = ['srn', 'gru', 'lstm']
    runs = ['1', '2', '3', '4', '5']

    # no
    for model in models:
        for run in runs:
            filename = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/nobrackets/logs/' + model + '_binaryfol_0brackets_' + run + '.txt'
            with open(filename, 'r') as f:
                for idx, line in enumerate(f):
                    if idx == 79:
                        test_acc = float(line.split('\t')[1])
                    elif idx == 81:
                        train_acc = float(line.split('\t')[1])
            data_list.append([model.upper(), 0, train_acc, test_acc])

    # half
    for model in models:
        for run in runs:
            filename = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/bracket_experiments/half_brackets/logs/' + model + '_binaryfol_halfbrackets_' + run + '.txt'
            with open(filename, 'r') as f:
                for idx, line in enumerate(f):
                    if idx == 29:
                        test_acc = float(line.split('\t')[1])
                    elif idx == 31:
                        train_acc = float(line.split('\t')[1])
            data_list.append([model.upper(), 50, train_acc, test_acc])

    # all
    for model in models:
        for run in runs:
            filename = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/final_experiments/binary_fol_rnn/bracket_experiments/all_brackets/logs/' + model + '_binaryfol_allbrackets_' + run + '.txt'
            with open(filename, 'r') as f:
                for idx, line in enumerate(f):
                    if idx == 29:
                        test_acc = float(line.split('\t')[1])
                    elif idx == 31:
                        train_acc = float(line.split('\t')[1])
            data_list.append([model.upper(), 100, train_acc, test_acc])

    df = pd.DataFrame(data_list, columns=cols)
    print(df)
    sns.set(font_scale=1.5,style="whitegrid")
    #sns.set(style="whitegrid")

    g = sns.factorplot(x="% brackets", y="testing accuracy", hue="model", data=df,
                       capsize=.05, palette="Blues", size=8,legend_out=False, aspect=1.5,ci='sd')
    plt.legend(loc='lower right')

    #plt.show()
    plt.savefig('factorplot_rnn_bracketexp.png',dpi=500)

get_bracket_data()