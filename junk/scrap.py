from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

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
#
# x = torch.rand(5, 3)
# y = torch.rand(5, 3)
# print(x)
# print(y)

# l_act = torch.ones(3)
# l_act = 0.5 * l_act
# r_act = torch.ones(3)
# print(l_act)
# print(r_act)
#
# kron = torch.ger(l_act, r_act).view(-1)
# print(kron)
# np.ravel(np.outer(l_act, r_act))
#

# a = Variable(torch.zeros(10))
# a[5] = 1
# print(a)
#
# print(torch.eye(10)[:,5])


def xavier_uniform_adapted(tensor, fan_in, fan_out, gain=1):
    """Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`U(-a, a)` where
    :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`.
    Also known as Glorot initialisation.
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    """
    if isinstance(tensor, Variable):
        xavier_uniform_adapted(tensor.data, fan_in, fan_out, gain=gain)
        return tensor

    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)

#TODO: write loop

#init.xavier_uniform(self.voc.bias)
# voc_fan_in = self.voc.weight.size(1)
# voc_fan_out = self.voc.weight.size(0)
# self.voc.weight = xavier_uniform_adapted(self.voc.weight, voc_fan_in, voc_fan_out)
# self.voc.bias = xavier_uniform_adapted(self.voc.bias, voc_fan_in, voc_fan_out)


# w = torch.Tensor(3, 5)
# nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
# print(w)
#
# v = torch.Tensor(1,7)
# nn.init.xavier_uniform(v)
# print(v)

# v = torch.Tensor(3,3).zero_()
# # v.view(1,7)
# # nn.init.xavier_uniform(v)
# print(v)
# fanin = v.size(1)
# fanout = v.size(0)
# new_v = xavier_uniform_adapted(v, fanin, fanout)
# print(new_v)
# new2_v = nn.init.xavier_uniform(v)
# print(new2_v)

a = torch.Tensor(3,3)
b = torch.Tensor(3,3)
print(torch.all(torch.eq(a,b)))
print(torch.all(torch.eq(a,a)))