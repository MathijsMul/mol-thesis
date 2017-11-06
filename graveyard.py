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
import logging
import datetime

##################################################################

logging.getLogger()
logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.info("Start time: %s" % datetime.datetime.now())

train_data_file = './data/junk/minitrain.txt'
test_data_file = train_data_file

tensors = False # tensors on or off (False -> tRNN, True -> tRNTN)

word_dim = 5 # dimensionality of word embeddings
cpr_dim = 5 # output dimensionality of comparison layer
num_epochs = 1 # number of epochs
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


net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,bound_layers=bound_layers, bound_embeddings=bound_embeddings)
net.load_state_dict(torch.load('models/tRNNminitrain.pt'))

final_acc = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=True)
print(final_acc)

