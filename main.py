from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datamanager as dat
from trnn import tRNN
from trntn import tRNTN
#import progressbar as pb
from test import compute_accuracy
import numpy as np
import random
import sys

##################################################################

# GLOBAL SETTINGS

# NL ANIMALS
# train_data_file = 'data/final/nl/nl_data1_animals_train.txt'
# test_data_file = 'data/final/nl/nl_data1_animals_test.txt'

# from command line:
# python3 main.py 'data/final/nl/nl_data1_animals_train.txt' 'data/final/nl/nl_data1_animals_test.txt' > nl_animals_date.txt

# FOL ANIMALS (translated from NL data)
# train_data_file = './data/final/fol/fol_animals_train_translated_from_nl.txt'
# test_data_file = './data/final/fol/fol_animals_test_translated_from_nl.txt'

# from command line:
# python3 main.py 'data/final/fol/fol_animals_train_translated_from_nl.txt' 'data/final/fol/fol_animals_test_translated_from_nl.txt' > fol_animals_date.txt

# FOL ANIMALS (new, do not use for now)
# train_data_file = './data/final/fol/fol_data1_animals_train.txt'
# test_data_file = './data/final/fol/fol_data1_animals_test.txt'

# FOL PEOPLE
# train_data_file = './data/final/fol/fol_data1_peopletrain.txt'
# test_data_file = './data/final/fol/fol_data1_peopletest.txt'

# from command line:
# python3 main.py 'data/final/fol/fol_data1_peopletrain.txt' 'data/final/fol/fol_data1_peopletest.txt' > fol_people_date.txt

# train_data_file = './data/minitrain.txt'
# test_data_file = train_data_file

train_data_file = './data/binary/split/binary1_train.txt'
test_data_file = './data/binary/split/binary1_test.txt'

# uncomment for execution from command line:
if __name__ == '__main__':
    train_data_file = sys.argv[1]
    test_data_file = sys.argv[2]
    tensors = sys.argv[3]

#tensors = False # tensors on or off (False -> tRNN, True -> tRNTN)

word_dim = 25 # dimensionality of word embeddings
cpr_dim = 75 # output dimensionality of comparison layer
num_epochs = 50 # number of epochs
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

if tensors:
    net = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
                bound_layers=bound_layers, bound_embeddings=bound_embeddings)
else:
    net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
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
print("Tensors on/off:        ", tensors)
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

acc_before_training = compute_accuracy(test_data, rels, net, print_outputs=False)
print("EPOCH", "\t", "ACCURACY")
print(str(0), '\t', str(acc_before_training))

##################################################################

# TRAINING

#print('Start training')

for epoch in range(num_epochs):  # loop over the dataset multiple times
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

#print('Finished Training \n')

##################################################################

# (FINAL) TESTING

final_acc = compute_accuracy(test_data, rels, net, print_outputs=False)
print(str(epoch + 1), '\t', str(final_acc))

#print('\n')