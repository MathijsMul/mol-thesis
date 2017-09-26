import torch

def show_accuracy(test_data, rels, net):
    correct = 0
    total = 0
    for i, data in enumerate(test_data.tree_data, 0):
        input, label = [[data[1], data[2]]], [rels.index(data[0])]
        label = torch.LongTensor(label)
        outputs = net(input)
        _, predicted = torch.max(outputs.data, 1)
        total += 1 # because test batch size is always 1
        correct += (predicted == label).sum()

    print('Accuracy of the network on the %d test images: %d %%' % (
        total, 100 * correct / total))