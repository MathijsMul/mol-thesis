import numpy as np
import matplotlib.pyplot as plt
import os

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
            if idx >= 25:
                acc = float(line.split()[1])
                acc_scores.append(acc)

    return(acc_scores)


#s = scrape_log('logs/trnn/fol_animals_1111.txt')

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

acc1 = scrape_log('./logs/binary/binary_neg_det1_trnn.txt')
acc2 = scrape_log('./logs/binary/binary_neg_det1_trntn.txt')
plot_two(acc1, acc2, 'Binary predicates, first quantifier negated', 'binary_neg_det1')


acc1 = scrape_log('./logs/binary/binary_neg_noun1_trnn.txt')
acc2 = scrape_log('./logs/binary/binary_neg_noun1_trntn.txt')
plot_two(acc1, acc2, 'Binary predicates, noun negated', 'binary_neg_noun')

acc1 = scrape_log('./logs/binary/binary_neg_verb_trnn.txt')
acc2 = scrape_log('./logs/binary/binary_neg_verb_trntn.txt')
plot_two(acc1, acc2, 'Binary predicates, verb negated', 'binary_neg_verb')

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