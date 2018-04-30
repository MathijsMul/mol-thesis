import torch
import numpy as np
import datamanager as dat
from torch.autograd import Variable
import os

# uncomment to generate confusion matrix in one go:
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import seaborn as sns

import pandas as pd
from collections import Counter, defaultdict

def compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=False, length_matrix=False, batch_size=32):
    # deactivate dropout
    net.eval()

    n_rels = len(rels)

    if confusion_matrix:
        batch_size=None
        confusion = torch.zeros(n_rels, n_rels)

    if length_matrix:
        batch_size=None
        length_all = torch.zeros(2,2)
        length_errors = torch.zeros(2,2)
        length_dict = {6 : 0, 9: 1}

    correct = 0.0
    total = 0

    if batch_size == None:
        for i, data in enumerate(test_data.tree_data, 0):

            input, label = [[data[1], data[2]]], [rels.index(data[0])]
            label = torch.LongTensor(label)
            outputs = net(input)
            _, predicted = torch.max(outputs.data, 1)

            if confusion_matrix:
                confusion[int(label[0])][int(predicted[0])] += 1

            if length_matrix:
                l1 = len(data[1])
                l2 = len(data[2])
                length_all[length_dict[l1]][length_dict[l2]] += 1
                #length_all = length_all.append({'length1': l1, 'length2': l2}, ignore_index=True)

            if print_outputs:
                print('Outputs:')
                print(outputs)
                print('Predicted label::')
                print(predicted)
                print('Real label:')
                print(label)
                if predicted.numpy() == label.numpy():
                    print('CORRECT')
                else:
                    print('FALSE')

            if predicted.numpy() == label.numpy():
                correct += 1
            elif length_matrix:
                length_errors[length_dict[l1]][length_dict[l2]] += 1

            total += 1 # because test batch size is always 1

        acc = 100 * correct / total
        acc = "%.2f" % round(acc, 2)

        if confusion_matrix:
            # create a confusion matrix, indicating for every actual relation (rows) which relation the network guesses (columns)

            print(confusion)
            print(confusion.sum())
            print('correct:')
            print(torch.trace(confusion))

            # normalize by dividing every row by its sum
            for i in range(n_rels):
                confusion[i] = confusion[i] / confusion[i].sum()

            # remove diagonal to focus on errors
            #for i in range(n_rels):
            #    confusion[i][i] = 0

            confusion = pd.DataFrame(confusion.numpy(), index=rels, columns=rels)
            h = sns.heatmap(confusion, cmap='Blues')
            h.set_yticklabels(rels, rotation=0)

            if False:
                # Set up plot
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(confusion.numpy(), cmap='hot')
                fig.colorbar(cax)

                # set up axes
                #ax.set_xticklabels([''] + rels, rotation=90)
                ax.set_xticklabels([''] + rels)
                ax.set_yticklabels([''] + rels)

                # force label at every tick
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

                # plt.show()

            plot_name = 'conf_' + net.__class__.__name__
            plt.savefig(plot_name)
            plt.close()

        if length_matrix:
            print('All test instances:')
            print(length_all)
            print('Errors:')
            print(length_errors)
    else:
        shuffle_samples = False
        batches = dat.BatchData(test_data, batch_size, shuffle_samples)
        batches.create_batches()

        total = 0
        for batch_idx in range(batches.num_batches):
            inputs = batches.batched_data[batch_idx]
            labels = batches.batched_labels[batch_idx]

            # convert label symbols to tensors
            labels = [rels.index(label) for label in labels]
            targets = torch.LongTensor(labels)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == targets).double().sum()
            total += len(inputs)

        acc = 100 * correct / total
        acc = "%.2f" % round(acc, 2)

    return(acc)

def comp_acc_per_length(test_data, rels, net, min_length, max_length, threed_plot):
    n_lengths = max_length - min_length + 1
    lengths = [i for i in range(min_length, max_length + 1)]

    if threed_plot:
        totals = torch.zeros(n_lengths, n_lengths)
        #corrects = torch.zeros(n_lengths, n_lengths)
        errors = torch.zeros(n_lengths, n_lengths)

    correct = 0.0
    total = 0
    total_errors = 0

    for i, data in enumerate(test_data.tree_data, 0):
        input, label = [[data[1], data[2]]], [rels.index(data[0])]
        length1 = len(data[1].leaves())
        length2 = len(data[2].leaves())
        label = torch.LongTensor(label)
        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)

        if (predicted == label).sum() == 1:
            correct += (predicted == label).sum()

        elif threed_plot:
            total_errors += 1
            # store errors
            errors[length1 - min_length][length2 - min_length] += 1

        total += 1 # because test batch size is always 1
        totals[length1 - min_length][length2 - min_length] += 1

    acc = 100 * correct / total
    acc = "%.2f" % round(acc, 2)

    if threed_plot:
        # create a confusion matrix, indicating for every actual relation (rows) which relation the network guesses (columns)

        # Normalize by dividing every cell (number of errors) by total number of length1, length2 occurrences
        # for i in range(n_lengths):
        #     for
        #     confusion[i] = confusion[i] / confusion[i].sum()

        #print(errors)
        errors = errors / totals
        #errors = errors / total_errors

        #corrects = corrects / totals
        # print(totals)
        # print(errors)

        # # Set up plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #cax = ax.matshow(corrects.numpy(), cmap='hot')
        cax = ax.matshow(errors.numpy(), cmap='hot')

        fig.colorbar(cax)

        # Set up axes
        #ax.set_xticklabels([''] + rels, rotation=90)
        ax.set_xticklabels([''] + lengths)
        ax.set_yticklabels([''] + lengths)

        # Force label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # sphinx_gallery_thumbnail_number = 2
        # plt.show()

        plot_name = 'length_errors_' + net.__class__.__name__

        plt.savefig(plot_name)

        plt.close()

    return(acc)

def comp_error_matrix(test_data, rels, net, show_totals):
    #os.environ['PATH'] = '/Library/TeX/texbin'

    totals = Counter()

    n_rels = len(rels)
    errors = {key : Counter() for key in rels}

    correct = 0.0
    total = 0

    for i, data in enumerate(test_data.tree_data, 0):
        # if i == 200:
        #    break

        input, label = [[data[1], data[2]]], [rels.index(data[0])]

        label = torch.LongTensor(label)
        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)

        label_symbol = rels[label.numpy()[0]]
        predicted_symbol = rels[predicted.numpy()[0]]

        if predicted.numpy() != label.numpy():

            errors[label_symbol][predicted_symbol] += 1

        totals[label_symbol] += 1
        correct += (predicted == label).sum()
        total += 1

    acc = 100 * correct / total
    acc = "%.2f" % round(acc, 2)

    # rels_to_tex = {'#': '\#', '<':'<', '=':'=', '>':'>', '^':'wedge', 'v':'lor', '|':'mid'}
    # print(errors)
    # print(errors.values())

    series = {}
    for rel in rels:
        #series[rels_to_tex[rel]] = [(0 if rel not in item else item[rel]) for item in errors.values()]
        series[rel] = [(0 if rel not in item else item[rel]) for item in errors.values()]

    #print(series)

    fig, ax = plt.subplots()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    ax.set_xticks(range(n_rels))
    #ax.set_xticklabels([rels_to_tex[rel] for rel in rels])
    ax.set_xticklabels(rels)

    bottom, x = np.zeros(n_rels), range(n_rels)

    if show_totals:
        cum_counts = [totals[rel] for rel in rels]
        ax.bar(x, cum_counts, width = 0.05, color='black')

    for key in series:
        # print(key)
        # print(series[key])
        # print(bottom)
        ax.bar(x, series[key], label=key, bottom=bottom)
        bottom += np.array(series[key])
    plt.legend()

    # for rel in rels:
    #     ax.bar(x, totals[rel], width=0.2)

    plt.title(net.__class__.__name__)
    #plt.show()
    plot_name = 'errorprofile_' + net.__class__.__name__

    if show_totals:
        plot_name += '_withtotals'

    plt.savefig(plot_name)