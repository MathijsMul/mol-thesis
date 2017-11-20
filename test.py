import torch

# uncomment to generate confusion matrix in one go:
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def compute_accuracy(test_data, rels, net, print_outputs, confusion_matrix=False):
    n_rels = len(rels)

    if confusion_matrix:
        confusion = torch.zeros(n_rels, n_rels)

    correct = 0.0
    total = 0

    for i, data in enumerate(test_data.tree_data, 0):
        input, label = [[data[1], data[2]]], [rels.index(data[0])]
        label = torch.LongTensor(label)
        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)

        if confusion_matrix:
            confusion[int(label[0])][int(predicted[0])] += 1

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

        correct += (predicted == label).sum()
        total += 1 # because test batch size is always 1

    acc = 100 * correct / total
    acc = "%.2f" % round(acc, 2)

    if confusion_matrix:
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

        plot_name = 'conf_' + net.__class__.__name__

        plt.savefig(plot_name)

        plt.close()

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

        # if i == 2:
        #     break

        if (predicted == label).sum() == 1:
            correct += (predicted == label).sum()
            #if threed_plot:
            #    corrects[length1 - min_length][length2 - min_length] += 1

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
