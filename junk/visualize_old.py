import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import torch
import seaborn as sns
import pandas as pd

def scrape_log_old(log_file):
    """
    scrapes log file to retrieve only accuracy scores

    :param log_file: log file containing development of accuracy score over epochs
    :return: list of accuracy scores
    """

    # list of accuracy scores for each epoch
    acc_scores = []
    with open(log_file, 'r') as f:
        final_line = None
        for idx, line in enumerate(f):
            if idx == 23:
                acc = float(line.split()[9])
                acc_scores.append(acc)

            if idx == 30:
                acc = float(line.split()[9])
                acc_scores.append(acc)

            elif idx > 30:
                if ((idx - 30) % 4) == 0:
                    split = line.split()
                    if len(split) > 0:
                        if not split[0] == 'Finished':
                            acc = float(split[9])
                            acc_scores.append(acc)
                        else:
                            final_line = idx + 2

                elif idx == final_line:
                    acc = float(line.split()[9])
                    acc_scores.append(acc)

    return(acc_scores)

def scrape_log(log_file):
    """
    scrapes log file to retrieve only accuracy scores

    :param log_file: log file containing development of accuracy score over epochs
    :return: list of accuracy scores
    """

    # list of accuracy scores for each epoch
    acc_scores = []

    with open(log_file, 'r') as f:
        final_line = None
        for idx, line in enumerate(f):
            if idx > 27 and idx < 79:
                acc = float(line.split()[1])
                acc_scores.append(acc)

    return(acc_scores)

def plot_all_models(root_dir):

    d = {}
    d['sumNN'] = scrape_log(root_dir + '/sumnn_f1_all_epochs.txt')
    d['tRNN'] = scrape_log(root_dir + '/trnn_f1_all_epochs.txt')
    d['tRNTN'] = scrape_log(root_dir + '/trntn_f1_all_epochs.txt')

    x = np.arange(0, 51)

    devs = pd.DataFrame({
        'sumNN': d['sumNN'],
        'tRNN': d['tRNN'],
        'tRNTN': d['tRNTN']}, index=x)
    print(devs)
    fig, ax = plt.subplots()

    devs.plot(ax=ax, color=sns.color_palette('Blues', 3))
    ax.set(xlabel='epoch',ylabel='accuracy')
    ax.legend()

    plt.savefig('bowman_rep_single_runs_acc_dev.png')


plot_all_models('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis')


def get_acc_dict(nl_animals, fol_animals, fol_people):
    d = {}
    d['NL animals'] = scrape_log(nl_animals)
    d['FOL animals'] = scrape_log(fol_animals)
    d['FOL people'] = scrape_log(fol_people)
    return(d)

def plot_single_acc(scores, net, name_plot):
    os.environ['PATH'] = '/Library/TeX/texbin'

    x = np.arange(0, len(scores))
    y = np.asarray(scores)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(x, y)

    plt.xlabel(r'\textbf{epoch}')
    plt.ylabel(r'\textit{accuracy}',fontsize=16)
    #plt.title(r"\TeX\ is Number "
    #          r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
    #          fontsize=16, color='gray')

    plt.title(r"Accuracy development " + net)

    # Make room for the ridiculously large title.
    plt.subplots_adjust(top=0.9)

    plt.savefig(name_plot)
    #plt.show()

def plot_two(acc1, acc2, name_plot, name_file):
    os.environ['PATH'] = '/Library/TeX/texbin'

    x = np.arange(0, len(acc1))
    y1 = np.asarray(acc1)
    y2 = np.asarray(acc2)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(x, y1, label="tRNN")
    plt.plot(x, y2, label="tRNTN")

    plt.xlabel(r'\textbf{epoch}')
    plt.ylabel(r'\textit{accuracy}',fontsize=16)

    #plt.title(r"Accuracy development binary data")
    plt.title(name_plot)

    # Make room for the ridiculously large title.
    plt.subplots_adjust(top=0.9)

    plt.legend()
    plt.savefig('acc_dev_' + name_file)

    plt.close()

#log1 ='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/logs/binary/2dets_4negs/default/rnns/adadelta/0_dropout/binary_2dets_4negs_gru_adad1.txt'
#log2 = log1
#acc1 = scrape_log(log1)
#acc2 = scrape_log('./logs/binary/binary_neg_det1_trntn.txt')
#plot_two(acc1, acc1, 'Binary predicates, first quantifier negated', 'binary_neg_det1')
#
#
# acc1 = scrape_log('./logs/binary/binary_neg_noun1_trnn.txt')
# acc2 = scrape_log('./logs/binary/binary_neg_noun1_trntn.txt')
# plot_two(acc1, acc2, 'Binary predicates, noun negated', 'binary_neg_noun')
#
# acc1 = scrape_log('./logs/binary/binary_neg_verb_trnn.txt')
# acc2 = scrape_log('./logs/binary/binary_neg_verb_trntn.txt')
# plot_two(acc1, acc2, 'Binary predicates, verb negated', 'binary_neg_verb')

def plot_dict_acc(acc_dict, net):
    os.environ['PATH'] = '/Library/TeX/texbin'

    x = np.arange(0, len(acc_dict['NL animals']))
    nl_animals = np.asarray(acc_dict['NL animals'])
    fol_animals = np.asarray(acc_dict['FOL animals'])
    fol_people = np.asarray(acc_dict['FOL people'])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(x, nl_animals, label="NL animals")
    plt.plot(x, fol_animals, label="FOL animals")
    plt.plot(x, fol_people, label="FOL people")

    plt.xlabel(r'\textbf{epoch}')
    plt.ylabel(r'\textit{accuracy}',fontsize=16)

    plt.title(r"Accuracy development " + net)

    # Make room for the ridiculously large title.
    plt.subplots_adjust(top=0.9)

    plt.legend()
    plt.savefig('acc_dev_' + net)
    #plt.show()
    plt.close()

# d_trnn = get_acc_dict('logs/trnn/nl_animals_1111.txt', 'logs/trnn/fol_animals_1111.txt', 'logs/trnn/fol_people_1111.txt')
# plot_dict_acc(d_trnn, 'tRNN')
#
# d_trntn = get_acc_dict('logs/trntn/nl_animals_1012.txt', 'logs/trntn/fol_animals_1012.txt', 'logs/trntn/fol_people_1012.txt')
# plot_dict_acc(d_trntn, 'tRNTN')

def confusion_matrix(test_data, rels, net, plot_name):
    n_rels = len(rels)

    confusion = torch.zeros(n_rels, n_rels)

    correct = 0.0
    total = 0

    for i, data in enumerate(test_data.tree_data, 0):
        input, label = [[data[1], data[2]]], [rels.index(data[0])]
        label = torch.LongTensor(label)
        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)

        #if confusion_matrix:
        confusion[int(label[0])][int(predicted[0])] += 1

        #correct += (predicted == label).sum()
        #total += 1 # because test batch size is always 1

    #acc = 100 * correct / total
    #acc = "%.2f" % round(acc, 2)

    # create a confusion matrix, indicating for every actual relation (rows) which relation the network guesses (columns)

    # Normalize by dividing every row by its sum
    for i in range(n_rels):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy(), cmap='hot')
    fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + rels, rotation=90)
    ax.set_xticklabels([''] + rels)
    ax.set_yticklabels([''] + rels)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    # plt.show()

    name = 'confusion_' + plot_name
    plt.savefig(name)

    plt.close()

    #return(acc)

def scatterplot_from_dict(data_dict):
    #for data_dict in d.values():
    x = data_dict.keys()
    y = [float(item) for item in data_dict.values()]

    plt.scatter(x, y)
    plt.xlabel('Percentage of brackets')
    plt.ylabel('Accuracy')
    plt.title('Test results with partial bracketing')

    #plt.legend(d.keys())
    plt.show()

#d = {0.0: '27.92', 0.1: '29.00', 0.2: '29.80', 0.3: '31.20', 0.4: '34.27', 0.5: '36.72', 0.6: '41.05', 0.7: '48.59', 0.8: '60.43', 0.9: '77.60', 1.0: '99.53'}

# for srn, trained on 2dets_4negs_train, tested on test version with increasing bracket ratios
#d = {0.0: '24.16', 0.1: '23.77', 0.2: '23.67', 0.3: '23.12', 0.4: '22.92', 0.5: '23.12', 0.6: '23.68', 0.7: '23.77', 0.8: '26.08', 0.9: '34.36', 1.0: '84.92'}

#scatterplot_from_dict(d)


#input:
# ( ( all warthogs ) walk )
#output:
# \Tree[.\text{( (all warthogs) walk )} [.\text{(all warthogs)} [ all warthogs ] ]  walk ]

def sentence_to_latex_tree(sentence):
    latex_tree = '\Tree['

    def make_node_name(node_name):
        out = '.\\' + 'text{' + node_name + '}'
        return(out)

    latex_tree += make_node_name(sentence)

    latex_tree += ']'
    return(latex_tree)

#print(sentence_to_latex_tree('( ( all warthogs ) walk )'))
