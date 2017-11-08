from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datamanager as dat
from trnn import tRNN
from trntn import tRNTN
from sumnn import sumNN
#import progressbar as pb
from test import compute_accuracy
import numpy as np
import random
import sys
import logging
import datetime

##################################################################

logging.getLogger()
logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.info("Start time: %s" % datetime.datetime.now())

# command line execution
if __name__ == '__main__':
    train_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    model = sys.argv[3]

# GLOBAL SETTINGS

word_dim = 25 # dimensionality of word embeddings
cpr_dim = 75 # output dimensionality of comparison layer
num_epochs = 5 # number of epochs
batch_size = 32 # Bowman takes 32
shuffle_samples = True
test_all_epochs = True # intermediate accuracy computation after each epoch
init_mode = 'xavier_uniform' # initialization of parameter weights
bound_layers = 0.05 # bound for uniform initialization of layer parameters
bound_embeddings = 0.01  # bound for uniform initialization of embeddings
l2_penalty = 1e-3 #customary: 2e-3 # weight_decay, l2 regularization term
save_params = False # store params at each epoch
show_progressbar = False
show_loss = False # show loss every 200 batches

##################################################################

# PREPARING DATA, NETWORK, LOSS FUNCTION AND OPTIMIZER

train_data = dat.SentencePairsDataset(train_data_file)
train_data.load_data(print_result=True)
vocab = train_data.word_list
rels = train_data.relation_list

test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

batches = dat.BatchData(train_data, batch_size, shuffle_samples)
batches.create_batches()

if model == 'tRNN':
    net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
                bound_layers=bound_layers, bound_embeddings=bound_embeddings)
elif model == 'tRNTN':
    net = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
               bound_layers=bound_layers, bound_embeddings=bound_embeddings)
elif model == 'sumNN':
    net = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
               bound_layers=bound_layers, bound_embeddings=bound_embeddings)

if save_params:
    params = {} # dictionary for storing params if desired
    params[0] = list(net.parameters())

# initialize parameters
net.initialize(mode=init_mode)
criterion = nn.NLLLoss()

optimizer = optim.Adadelta(net.parameters(), weight_decay = l2_penalty)

##################################################################
# print hyperparameters

print("\n")
print("MODEL SETTINGS")
print("Model:                 ", model)
print("Train data:            ", train_data_file)
print("Test data:             ", test_data_file)
print("Num. epochs:           ", num_epochs)
print("Word dim.:             ", word_dim)
print("Cpr. dim.:             ", cpr_dim)
print("Batch size:            ", batch_size)
print("Shuffle samples:       ", shuffle_samples)
print("Weight initialization: ", init_mode)
print("Optimizer:             ", optimizer.__class__.__name__)
print("L2 penalty:            ", l2_penalty)
print("Num. train instances:  ", len(train_data.tree_data))
print("Num. test instances:   ", len(test_data.tree_data))
print("\n")

##################################################################

acc_before_training = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=False)
print("EPOCH", "\t", "ACCURACY")
print(str(0), '\t', str(acc_before_training))

##################################################################

# TRAINING

#print('Start training')

for epoch in range(num_epochs):  # loop over the dataset multiple times
    logging.info("Training epoch %i" % (epoch + 1))

    #print('EPOCH ', str(epoch + 1))
    running_loss = 0.0

    if show_progressbar:
        bar = pb.ProgressBar(max_value=batches.num_batches)

    # shuffle at each epoch
    if shuffle_samples and epoch > 0:
        batches = dat.BatchData(train_data, batch_size, shuffle_samples)
        batches.create_batches()

    for i in range(batches.num_batches):
        if show_progressbar:
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

        # print loss statistics
        if show_loss:
            running_loss += loss.data[0]
            if i % 100 == 1:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 100, running_loss / 100))
                running_loss = 0.0

        #bar.update(i + 1)

    if show_progressbar:
        bar.finish()
        #print('\n')

    if test_all_epochs and epoch < (num_epochs - 1):
        acc = compute_accuracy(test_data, rels, net, print_outputs=False)
        print(str(epoch + 1), '\t', str(acc))

    if save_params:
        params[epoch + 1] = list(net.parameters())

##################################################################

# SAVING AND FINAL TESTING

# save model
model_name = net.__class__.__name__ + train_data_file.split('/')[-1].split('.')[0] + '.pt'
torch.save(net.state_dict(), 'models/' + model_name)

final_acc = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=False)
print(str(epoch + 1), '\t', str(final_acc))

logging.info("End time: %s" % datetime.datetime.now())
