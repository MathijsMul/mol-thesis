from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class sumNN(nn.Module):
    """
    a summing NN baseline which is largely identical to the TreeRNN, except that instead of
    using a learned composition function, it simply sums the term vectors in each expression to compose
    them before passing them to the comparison layer
    """

    def __init__(self, vocab, rels, word_dim, cpr_dim, bound_layers, bound_embeddings):
        super().__init__()
        self.word_dim = word_dim # dimensionality of word embeddings
        self.cpr_dim = cpr_dim # output dimensionality of comparison layer
        self.num_rels = len(rels) # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.word_dim, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {word: i for i, word in enumerate(vocab)}

        # activation functions
        self.relu = nn.LeakyReLU() # cpr layer, negative slope is 0.01, which is standard

        self.log_softmax = nn.LogSoftmax()

        self.bound_layers = bound_layers
        self.bound_embeddings = bound_embeddings

    def initialize(self, mode):
        """
        Initialization of parameters

        :return:
        """

        # always initialize biases as zero vectors:
        self.cpr.bias.data.fill_(0)
        self.sm.bias.data.fill_(0)

        if mode == 'xavier_uniform':
            # much beter results
            init.xavier_uniform(self.voc.weight)
            init.xavier_uniform(self.cpr.weight, gain = math.sqrt(2/(1 + (0.01**2)))) # rec. gain for leakyrelu
            init.xavier_uniform(self.sm.weight)

        if mode == 'xavier_normal':
            init.xavier_normal(self.voc.weight)
            init.xavier_normal(self.cpr.weight, gain = math.sqrt(2/(1 + (0.01**2))))
            init.xavier_normal(self.sm.weight)

        if mode == 'uniform':
            init.uniform(self.voc.weight, -1*self.bound_embeddings, self.bound_embeddings)
            init.uniform(self.cpr.weight, -1*self.bound_layers, self.bound_layers)
            init.uniform(self.sm.weight, -1*self.bound_layers, self.bound_layers)

    def make_sentence_vector(self, sentence):
        idxs = [self.word_dict[word] for word in sentence]
        tensor = torch.LongTensor(idxs)
        tensor_var = Variable(tensor)
        embedding = self.voc(tensor_var)

        # composition step is just vector addition
        summed_embeddings = torch.sum(embedding, 0)
        return(summed_embeddings)

    def make_batch_matrix(self, batch, size_batch):

        # make container
        batch_out = Variable(torch.zeros((1, size_batch, self.word_dim)))

        for i in range(size_batch):
            sentence_vector = self.make_sentence_vector(batch[i])
            batch_out[:,i] = sentence_vector
        return(batch_out)

    def forward(self, inputs):
        """

        :param inputs: list of lists of form [left_tree, right_tree], to support minibatch of size > 1
        :return: outputs, tensor of dimensions batch_size x num_classes
        """
        size_batch = len(inputs)

        left_inputs = [input[0] for input in inputs]
        right_inputs = [input[1] for input in inputs]
        left_sums = self.make_batch_matrix(left_inputs, size_batch)
        right_sums = self.make_batch_matrix(right_inputs, size_batch)

        concat = torch.cat((left_sums, right_sums), 2)  # dimensions 1 x batch_size x word_dim
        apply_cpr = self.cpr(concat)
        cpr_activated = self.relu(apply_cpr)
        to_softmax = self.sm(cpr_activated)
        to_softmax = to_softmax.view(size_batch, self.num_rels) # must be of size N x C
        outputs = self.log_softmax(to_softmax) # take log for NLLLoss

        return(outputs) # size: batch_size x num_classes

