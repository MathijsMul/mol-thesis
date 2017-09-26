from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#
# t = torch.rand(4,7)
# print(t)
#
# t[1,:] = torch.ones(1,7)
# print(t)


# import datamanager as dat
# data = dat.SentencePairsDataset(data_file = 'data/fol_datasmall_people_train.txt')
# data.load_data()
# batches = dat.BatchData(data, 4, False)
# net = tRNN(data.word_list, data.relation_list, 25, 75)
#
# for i, data in enumerate(batches.create_batches()):
#     inputs, labels = data
#     #labels = [Variable(torch.LongTensor([rels.index(label)])) for label in labels]
#     net(inputs)