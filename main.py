from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datamanager as dat
from trnn import tRNN
import progressbar as pb

##################################################################

# SETTINGS

train_data_file = 'data/fol_data1_animals_train.txt'
test_data_file = 'data/fol_data1_animals_test.txt'
word_dim = 25
cpr_dim = 75
num_epochs = 10
batch_size = 32
shuffle_samples = False

##################################################################

# PREPARING DATA, NETWORK, LOSS FUNCTION AND OPTIMIZER

train_data = dat.SentencePairsDataset(train_data_file)
train_data.load_data()
batches = dat.BatchData(train_data, batch_size, shuffle_samples)
batches.create_batches()
vocab = train_data.word_list
rels = train_data.relation_list

net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim)

criterion = nn.NLLLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adadelta(net.parameters(), lr=0.001) # not sure about this
optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()),
                                       lr=0.2, weight_decay=0.1)

##################################################################

# TRAINING

print('Start training')

for epoch in range(num_epochs):  # loop over the dataset multiple times
    print('EPOCH ', str(epoch + 1))
    running_loss = 0.0
    bar = pb.ProgressBar(max_value=batches.num_batches)
    for i in range(batches.num_batches):
        bar.update(i)

        inputs = batches.batched_data[i]
        labels = batches.batched_labels[i]

        # convert label symbols to tensors
        labels = [rels.index(label) for label in labels]
        targets = Variable(torch.LongTensor(labels))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        bar.update(i + 1)
    bar.finish()
    print('\n')

print('Finished Training \n')


##################################################################

# TESTING

test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

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

