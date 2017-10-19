import torch

def compute_accuracy(test_data, rels, net, print_outputs):
    correct = 0.0
    total = 0
    for i, data in enumerate(test_data.tree_data, 0):
        input, label = [[data[1], data[2]]], [rels.index(data[0])]
        label = torch.LongTensor(label)
        #print('test result')
        outputs = net(input)

        _, predicted = torch.max(outputs.data, 1)
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
        #correct += torch.eq(predicted, label)
        total += 1 # because test batch size is always 1

    acc = 100 * correct / total
    acc = "%.2f" % round(acc, 2)
    #
    # print('Accuracy of the network on the %d test instances: %.2f %%' % (total, acc))

    return(acc)