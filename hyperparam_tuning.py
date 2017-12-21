from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datamanager as dat
from trnn import tRNN
from trntn import tRNTN
from sumnn import sumNN
from srn import SRN
from gru import GRU
from lstm import LSTM
from test import compute_accuracy
import sys

# GLOBAL SETTINGS

# train_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
# test_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_test.txt'

train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_train.txt'
test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_test.txt'

model = 'sumNN'
num_epochs = 50
word_dim = 25 # dimensionality of word embeddings
cpr_dim = 75 # output dimensionality of comparison layer
batch_size = 32 # Bowman takes 32
shuffle_samples = True
test_all_epochs = False # intermediate accuracy computation after each epoch
init_mode = 'xavier_uniform' # initialization of parameter weights
bound_layers = 0.05 # bound for uniform initialization of layer parameters
bound_embeddings = 0.01  # bound for uniform initialization of embeddings
#l2_penalty = 1e-3 #customary: 2e-3 # weight_decay, l2 regularization term
        # (Michael: tensor model: 2e-3; matrix model: 2e-4,Bowman: matrix: 0.001, tensor: 0.0003)
show_loss = False # show loss every 100 batches
sequential_loading = (model not in ['tRNN', 'tRNTN'])
n_hidden = 128

train_data = dat.SentencePairsDataset(train_data_file)
train_data.load_data(sequential=sequential_loading, print_result=True)
vocab = train_data.word_list
rels = train_data.relation_list

test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data(sequential=sequential_loading)

criterion = nn.NLLLoss()

# batch data
batches = dat.BatchData(train_data, batch_size, shuffle_samples)
batches.create_batches()

#2e-3; matrix model: 2e-4,Bowman: matrix: 0.001, tensor: 0.0003
l2_options = [3e-4]

for l2_penalty in l2_options:

    if model == 'tRNN':
        net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
                   bound_layers=bound_layers, bound_embeddings=bound_embeddings)
    elif model == 'tRNTN':
        net = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
                    bound_layers=bound_layers, bound_embeddings=bound_embeddings)
    elif model == 'sumNN':
        net = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
                    bound_layers=bound_layers, bound_embeddings=bound_embeddings)
    elif model == 'SRN':
        net = SRN(vocab, rels, word_dim, n_hidden, cpr_dim)
    elif model == 'GRU':
        net = GRU(vocab, rels, word_dim, n_hidden, cpr_dim)
    elif model == 'LSTM':
        net = LSTM(vocab, rels, word_dim, n_hidden, cpr_dim)

    if init_mode:
        # initialize parameters according to preferred mode, if provided
        net.initialize(mode=init_mode)

    print(net)
    print('l2_penalty: %s' % str(l2_penalty))
    optimizer = optim.Adadelta(net.parameters(), weight_decay=l2_penalty)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Training epoch %i" % (epoch + 1))
        # shuffle at each epoch
        if shuffle_samples and epoch > 0:
            batches = dat.BatchData(train_data, batch_size, shuffle_samples)
            batches.create_batches()

        # if epoch < (num_epochs - 1):
        #     acc = compute_accuracy(test_data, rels, net, print_outputs=False)
        #     print(str(epoch + 1), '\t', str(acc))

        for i in range(batches.num_batches):
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

            # print loss statistics
            if show_loss:
                running_loss += loss.data[0]
                if (i + 2) % 100 == 1:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i, running_loss / 100))
                    running_loss = 0.0

    final_acc = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=False)
    print(str(epoch + 1), '\t', str(final_acc))


