from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from diagnostic_datamanager import DiagnosticDataFile
import numpy as np
from collections import Counter, defaultdict
import pickle
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from rnn import RNN
import pandas as pd
import seaborn as sns

def load_model(model_path):
    #model_path = 'linreg_length.pkl'
    diag_classifier_pkl = open(model_path, 'rb')
    diag_classifier = pickle.load(diag_classifier_pkl)
    return(diag_classifier)

def data_distribution(data):
    labels = [item[0].numpy()[0] for item in data]
    total = len(labels)
    counter = Counter(labels)
    counter_normalized = {label : counter[label] / total for label in counter.keys()}
    counter_out = {label : "%.2f" % round(counter_normalized[label], 2) for label in counter_normalized.keys()}
    return(counter_out)

def compute_mse(hypotheses, predictions):
    mse = metrics.mean_squared_error(hypotheses, predictions)
    return (mse)

def compute_mae(hypotheses, predictions):
    mae = metrics.mean_absolute_error(hypotheses, predictions)
    return(mae)

def compute_acc(hypotheses, predictions):
    # for continuous outputs:
    # train_hyps = np.array([item[0].float().numpy() for item in train_data]).ravel()

    try:
        acc = metrics.accuracy_score(hypotheses, predictions)
    except:
        # round predictions
        pred_rounded = np.rint(predictions)
        acc = metrics.accuracy_score(hypotheses, pred_rounded)
    return (acc)

def plot_predictions(hypotheses, predictions, title, show_boxplot=True, show_violinplot=False, show_confmatrix=True):
    plt.plot(hypotheses, hypotheses, color='blue', linewidth=2)

    if show_boxplot or show_violinplot:
        box_data = defaultdict(list)

        for idx in range(len(hypotheses) - 1):
            true = hypotheses[idx]
            pred = predictions[idx]
            box_data[true] += [pred]

        box_data = dict(box_data)
        data = [np.array(preds) for preds in box_data.values()]
        pos = [np.array(position) for position in box_data.keys()]

        if show_boxplot:
            plt.boxplot(data, positions=pos, showfliers=False)
            title += 'box'
        if show_violinplot:
            plt.violinplot(data,
                   showmeans=False,
                   showmedians=True)
            title += 'violin'

        plt.xticks()
        plt.yticks()
        plt.title(title)

        plt.savefig(title)
        plt.close()

    if show_confmatrix:
        # for pos tags:
        #pos_labels = ['quant', 'noun', 'verb', 'neg']
        #mon_dirs = ['downward', 'neutral', 'upward']
        neg_options = ['unnegated', 'negated']

        conf = metrics.confusion_matrix(hypotheses, predictions)
        # normalize
        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

        #confusion = pd.DataFrame(conf, index=pos_labels, columns=pos_labels)
        #confusion = pd.DataFrame(conf, index=mon_dirs, columns=mon_dirs)
        confusion = pd.DataFrame(conf, index=neg_options, columns=neg_options)

        h = sns.heatmap(confusion, cmap='Blues')
        h.set_yticklabels(neg_options, rotation=0)
        plot_name = 'conf_' + title
        plt.savefig(plot_name)
        plt.close()

def train(train_data, classifier, hypothesis, print_train_distribution=True, plot_pred=True):

    if print_train_distribution:
        print('Train distribution:')
        print(data_distribution(train_data))

    if classifier == 'logreg':
        diag_classifier = LogisticRegression()
    elif classifier == 'linreg':
        diag_classifier = LinearRegression()

    train_hiddens = np.array([item[1].numpy() for item in train_data])
    train_hyps = np.array([item[0].numpy() for item in train_data]).ravel()
    print(train_hyps)

    diag_classifier.fit(train_hiddens, train_hyps)

    train_pred = diag_classifier.predict(train_hiddens)

    print('Training MSE score:')
    print(compute_mse(train_hyps, train_pred))
    print('Training MAE score:')
    print(compute_mae(train_hyps, train_pred))
    print('Training accuracy:')
    print(compute_acc(train_hyps, train_pred))

    plot_name = 'Training results ' + hypothesis
    if plot_pred:
        plot_predictions(train_hyps, train_pred, plot_name)

    # save model
    model_path = classifier + '_' + hypothesis + '.pkl'
    model_pkl = open(model_path, 'wb')
    pickle.dump(diag_classifier, model_pkl)
    model_pkl.close()
    return(diag_classifier)

def test(test_data, diag_classifier, hypothesis, plot=True):
    test_hiddens = np.array([item[1].numpy() for item in test_data])
    test_hyps = np.array([item[0].numpy() for item in test_data]).ravel()
    test_pred = diag_classifier.predict(test_hiddens)

    print('Test distribution:')
    print(data_distribution(test_data))

    print('Testing MSE score:')
    print(compute_mse(test_hyps, test_pred))
    print('Testing MAE score:')
    print(compute_mae(test_hyps, test_pred))
    print('Testing accuracy:')
    print(compute_acc(test_hyps, test_pred))

    plot_name = 'Testing results ' + hypothesis

    if plot:
        plot_predictions(test_hyps, test_pred, plot_name)