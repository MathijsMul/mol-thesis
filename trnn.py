from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class tRNN(nn.Module):
    """
    tree-shaped recurrent neural network
    """

    def __init__(self, vocab, rels, word_dim, cpr_dim, bound_layers, bound_embeddings):
        super().__init__()
        self.word_dim = word_dim # dimensionality of word embeddings
        self.cpr_dim = cpr_dim # output dimensionality of comparison layer
        self.num_rels = len(rels) # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        # composition matrix
        self.cps = nn.Linear(2 * self.word_dim, self.word_dim)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.word_dim, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {}
        for word in vocab:
            # create one-hot encodings for words in vocabulary
            # self.word_dict[word] = Variable(torch.eye(self.voc_size)[:,vocab.index(word)], requires_grad=True)
            self.word_dict[word] = Variable(torch.LongTensor([vocab.index(word)])) #.view(-1)

        # activation functions
        self.relu = nn.LeakyReLU() # cpr layer, negative slope is 0.01, which is standard
        self.tanh = nn.Tanh() # cps layers

        self.bound_layers = bound_layers
        self.bound_embeddings = bound_embeddings

    def initialize(self, mode):
        """
        Initialization of parameters

        :return:
        """

        # always initialize biases as zero vectors:
        self.cps.bias.data.fill_(0)
        self.cpr.bias.data.fill_(0)
        self.sm.bias.data.fill_(0)

        if mode == 'xavier_uniform':
            # much beter results
            init.xavier_uniform(self.voc.weight)
            init.xavier_uniform(self.cps.weight, gain = 5/3) # recommended gain for tanh
            init.xavier_uniform(self.cpr.weight, gain = math.sqrt(2/(1 + (0.01**2)))) # rec. gain for leakyrelu
            init.xavier_uniform(self.sm.weight)

        if mode == 'xavier_normal':
            init.xavier_normal(self.voc.weight)
            init.xavier_normal(self.cps.weight, gain = 5/3)
            init.xavier_normal(self.cpr.weight, gain = math.sqrt(2/(1 + (0.01**2))))
            init.xavier_normal(self.sm.weight)

        if mode == 'uniform':
            init.uniform(self.voc.weight, -1*self.bound_embeddings, self.bound_embeddings)
            init.uniform(self.cps.weight, -1*self.bound_layers, self.bound_layers)
            init.uniform(self.cpr.weight, -1*self.bound_layers, self.bound_layers)
            init.uniform(self.sm.weight, -1*self.bound_layers, self.bound_layers)

    def forward(self, inputs):
        """

        :param inputs: list of lists of form [left_tree, right_tree], to support minibatch of size > 1
        :return: outputs, tensor of dimensions batch_size x num_classes
        """

        # handles batch with multiple inputs, inserted as list
        outputs = Variable(torch.rand(len(inputs), self.num_rels))

        for idx, input in enumerate(inputs):
            left = input[0]
            right = input[1]
            left_cps = self.compose(left)
            right_cps = self.compose(right)
            apply_cpr = self.cpr(torch.cat((left_cps, right_cps)))
            activated_cpr = self.relu(apply_cpr)
            to_softmax = self.sm(activated_cpr).view(1, self.num_rels)
            # NLL loss function requires log probabilities! so must use log_softmax here instead of softmax:
            output = F.log_softmax(to_softmax)
            outputs[idx,:] = output
        return(outputs) # size: batch_size x num_classes

    def compose(self, tree):
        if tree.label() == '.': # leaf nodes: get word embedding
            # word_onehot = self.word_dict[tree[0]]
            # emb = self.voc(word_onehot)

            embedded = self.voc(self.word_dict[tree[0]]).view(-1)
            return(embedded)

        else:
            concat = torch.cat((self.compose(tree[0]), self.compose(tree[1])))
            cps = self.cps(concat)
            activated_cps = self.tanh(cps)
            return activated_cps


