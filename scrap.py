from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import datamanager as dat
from trnn import tRNN
from trntn import tRNTN
import progressbar as pb
from test import show_accuracy
import numpy as np
import random

##################################################################

# GLOBAL SETTINGS

tensors = False # tensors on or off
# train_data_file = 'data/fol_datasmall_people_train.txt'
# # test_data_file = 'data/fol_datasmall_people_test.txt'
train_data_file = 'data/nl_data1_animals_train.txt'
test_data_file = 'data/nl_data1_animals_test.txt'
# train_data_file = 'data/mini2.txt'
# test_data_file = 'data/mini2.txt'
# train_data_file = 'data/minitrain.txt'
# test_data_file = train_data_file
word_dim = 25 # dimensionality of word embeddings
cpr_dim = 75 # output dimensionality of comparison layer
num_epochs = 100
batch_size = 32
shuffle_samples = True
test_all_epochs = True # intermediate accuracy computation after each epoch
# bound_layers = 3 # bound for uniform initialization of layer parameters
# bound_embeddings = 2  # bound for uniform initialization of embeddings
bound_layers = 1 # bound for uniform initialization of layer parameters
bound_embeddings = 3  # bound for uniform initialization of embeddings
#learning_rate = 0.01
l2_penalty = 2e-3 # weight_decay, l2 regularization term

save_params = False # store params at each epoch
show_progressbar = False

##################################################################

# PREPARING DATA, NETWORK, LOSS FUNCTION AND OPTIMIZER

train_data = dat.SentencePairsDataset(train_data_file)
train_data.load_data()
batches = dat.BatchData(train_data, batch_size, shuffle_samples)
batches.create_batches()
vocab = train_data.word_list
rels = train_data.relation_list
#print(rels)
#rels = ['>', '=']

test_data = dat.SentencePairsDataset(test_data_file)
test_data.load_data()

if tensors:
    net = tRNTN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
                bound_layers=bound_layers, bound_embeddings=bound_embeddings)
else:
    net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
               bound_layers=bound_layers, bound_embeddings=bound_embeddings)

# if save_params:
#     params = {} # dictionary for storing params if desired
#     params[0] = list(net.parameters())

# uniformly initialize parameters for chosen range
# net.initialize()

#print(list(net.parameters()))

#criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
#
# # TODO: check optimizer
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=l2_penalty)

# got effect for this setting:
# optimizer = optim.SGD(net.parameters(), lr=0.01)

#optimizer = optim.Adadelta(net.parameters(), lr=0.001)
#optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()),
#                                       lr=0.2, weight_decay=0.1)


# negative log likelihood
criterion = nn.NLLLoss()

# TODO: check optimizer
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=l2_penalty)

optimizer = optim.Adagrad(net.parameters(), weight_decay=l2_penalty)
# got effect
##################################################################

# for idx, data in enumerate(train_data.tree_data):
#     output = net([[data[1], data[2]]])
#     print(output)
#     if idx == 100:
#         break

# for bound_layers in [0.01, 0.02, 0.03, 0.04, 0.05]:
#     for bound_embeddings in [0.01, 0.02, 0.03, 0.04, 0.05]:
#         print('layer bound:')
#         print(bound_layers)
#         print('embeddings bound:')
#         print(bound_embeddings)
#         net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
#                bound_layers=bound_layers, bound_embeddings=bound_embeddings)
#
#         net.initialize()
#         # for idx, data in enumerate(train_data.tree_data):
#         #     # print(torch.max(output.data, 1))
#         #     if idx%100 == 0:
#         #         output = net([[data[1], data[2]]])
#         #         print(output)
#         for epoch in range(2):
#             for i in range(batches.num_batches):
#
#                 inputs = batches.batched_data[i]
#                 labels = batches.batched_labels[i]
#
#                 # convert label symbols to tensors
#                 labels = [rels.index(label) for label in labels]
#                 #print('targets')
#                 #print(labels)
#                 #labels = [0 if random.random()<0.5 else 1 for i in range(len(labels))]
#                 targets = Variable(torch.LongTensor(labels))
#                 #print('targets')
#                 #print(targets)
#
#                 #a = list(net.parameters())[0].clone()
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 #net.zero_grad()
#
#                 # forward + backward + optimize
#                 outputs = net(inputs)
#                 #print('outputs')
#                 #print(outputs)
#
#                 loss = criterion(outputs, targets)
#                 #print(loss)
#                 loss.backward()
#                 optimizer.step()
#
#         show_accuracy(test_data, rels, net, print_outputs=False)

#
#
# for i in range(10):
#     net = tRNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim,
#                bound_layers=bound_layers, bound_embeddings=bound_embeddings)
#     net.initialize()
#
#     for idx, data in enumerate(train_data.tree_data):
#
#
#         # print(torch.max(output.data, 1))
#         if idx%100 == 0:
#             output = net([[data[1], data[2]]])
#             print(output)

#
# dat1 = train_data.tree_data[0]
# print(dat1)
# dat2 = train_data.tree_data[999]
# print(dat2)
#
# net2 = net
#
# output1 = net([[dat1[1], dat1[2]]])
# output2 = net2([[dat2[1], dat2[2]]])
#
# print(output1, output2)

# print('COMPARE OUTPUTS')
# i = 0
# for ins in train_data.tree_data:
#
#     if i < 5:
#         left, right = ins[1], ins[2]
#         print(left)
#         print(right)
#         out = net([left, right])
#         print(out)
#         i += 1


#print(params)
#
# def xavier_uniform_adapted(tensor, fan_in, fan_out, gain=1):
#     """Fills the input Tensor or Variable with values according to the method
#     described in "Understanding the difficulty of training deep feedforward
#     neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
#     distribution. The resulting tensor will have values sampled from
#     :math:`U(-a, a)` where
#     :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`.
#     Also known as Glorot initialisation.
#     Args:
#         tensor: an n-dimensional torch.Tensor or autograd.Variable
#         gain: an optional scaling factor
#
#     """
#     if isinstance(tensor, Variable):
#         xavier_uniform_adapted(tensor.data, fan_in, fan_out, gain=gain)
#         return tensor
#
#     std = gain * math.sqrt(2.0 / (fan_in + fan_out))
#     a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
#            return tensor.uniform_(-a, a

# import fol_gen as fg
#
# nl_file = 'data/final/nl/nl_data1_animals_train.txt'
# fol_file = 'folresult1.txt'
#
# def extend_sentence(split_sentence):
#     if split_sentence[3] != '(':
#         split_sentence = split_sentence[:3] + ['(', ''] + [split_sentence[3]] + [')'] + split_sentence[4:]
#     if split_sentence[8] != '(':
#         split_sentence = split_sentence[:8] + ['(', ''] + [split_sentence[8]] + [')'] + split_sentence[9:]
#     return(split_sentence)
#
# with open(nl_file, 'r') as nl_f:
#     for idx, line in enumerate(nl_f):
#         all = line.split('\t')
#         left = all[1]
#         right = all[2]
#
#         l = extend_sentence(left.split())
#         premise = [l[2], l[4], l[5], l[9], l[10]]
#         r = extend_sentence(right.split())
#         hypothesis = [r[2], r[4], r[5], r[9], r[10]]
#
#         print(premise)
#         print(hypothesis)
#
#         filtered_axioms = fg.filter_axioms(fg.axioms, l[2], l[4], l[5], l[9], l[10], r[2], r[4], r[5], r[9], r[10])
#
#         rel = fg.interpret(s, filtered_axioms)
#         training_file.write(matlab_string(d) + "\n")
#
#         if idx == 15:
#             break
# )
#
# class BatchData():
#     def __init__(self, unbatched_data, batch_size, shuffle):
#         self.unbatched_data = unbatched_data.tree_data
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_samples = len(unbatched_data)
#         self.batched_data = []
#         self.batched_labels = []
#         self.num_batches = 0
#
#     def create_batches(self):
#         if self.shuffle:
#             indices = list(range(self.num_samples))
#             np.random.shuffle(indices)
#             #print(indices)
#
#         last_idx = self.num_samples - (self.num_samples % self.batch_size)
#         #print(last_idx)
#         #for start_idx in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
#         #print(range(0, last_idx, self.batch_size))
#
#         for start_idx in range(0, last_idx + 1, self.batch_size):
#             print('idx')
#             print(start_idx)
#             if self.shuffle:
#                 #batch = indices[start_idx:start_idx + self.batch_size]
#                 #batch = slice(indices[start_idx], indices[start_idx] + self.batch_size)
#                 if start_idx == last_idx:
#                     batch = [indices[i] for i in range(start_idx, self.num_samples - 1)]
#                 else:
#                     batch = [indices[i] for i in range(start_idx, start_idx + self.batch_size)]
#                 #print(batch)
#                 data = []
#                 labels = []
#                 for idx in batch:
#                     sample = self.unbatched_data[idx]
#                     data.append([sample[1], sample[2]])
#                     labels.append(sample[0])
#                 self.batched_data.append(data)
#                 self.batched_labels.append(labels)
#
#             else:
#                 batch = slice(start_idx, start_idx + self.batch_size)
#                 self.batched_data.append([[sample[1], sample[2]] for sample in self.unbatched_data[batch]])
#                 self.batched_labels.append([sample[0] for sample in self.unbatched_data[batch]])
#
#             #print(batch)
#             # alternatively with a generator, but this is probably slower?
#             #yield [[sample[1], sample[2]] for sample in self.unbatched_data[batch]],\
#             #      [sample[0] for sample in self.unbatched_data[batch]]
#
#             self.num_batches += 1