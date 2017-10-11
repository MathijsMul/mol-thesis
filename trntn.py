from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class tRNTN(nn.Module):
    """
    tree-shaped recurrent neural tensor network
    """

    def __init__(self, vocab, rels, word_dim, cpr_dim, bound_layers, bound_embeddings):
        super().__init__()
        self.word_dim = word_dim # dimensionality of word embeddings
        self.cpr_dim = cpr_dim # output dimensionality of comparison layer
        self.num_rels = len(rels) # number of relations (labels)
        self.voc_size = len(vocab)

        # vocabulary matrix
        self.voc = nn.Linear(self.voc_size, self.word_dim)

        # composition matrix
        self.cps = nn.Linear(2 * self.word_dim, self.word_dim)

        # composition tensor
        self.cps_t = nn.Linear(self.word_dim * self.word_dim, self.word_dim)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.word_dim, self.cpr_dim)

        # comparison tensor
        self.cpr_t = nn.Linear(self.word_dim * self.word_dim, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {}
        for word in vocab:
            # create one-hot encodings for words in vocabulary
            self.word_dict[word] = Variable(torch.zeros(self.voc_size))
            self.word_dict[word][vocab.index(word)] = 1

        # activation functions
        self.relu = nn.LeakyReLU() # cpr layer, negative slope is 0.01, which is standard
        self.tanh = nn.Tanh() # cps layers

        self.bound_layers = bound_layers
        self.bound_embeddings = bound_embeddings

    def initialize(self):
        """
        Uniform weight initialization

        :return:
        """

        init.uniform(self.voc.weight, -1*self.bound_embeddings, self.bound_embeddings)
        init.uniform(self.voc.bias, -1*self.bound_embeddings, self.bound_embeddings)

        init.uniform(self.cps.weight, -1*self.bound_layers, self.bound_layers)
        init.uniform(self.cps.bias, -1*self.bound_layers, self.bound_layers)

        init.uniform(self.cps_t.weight, -1*self.bound_layers, self.bound_layers)
        init.uniform(self.cps_t.bias, -1*self.bound_layers, self.bound_layers)

        init.uniform(self.cpr.weight, -1*self.bound_layers, self.bound_layers)
        init.uniform(self.cpr.bias, -1*self.bound_layers, self.bound_layers)

        init.uniform(self.cpr_t.weight, -1*self.bound_layers, self.bound_layers)
        init.uniform(self.cpr_t.bias, -1*self.bound_layers, self.bound_layers)

        init.uniform(self.sm.weight, -1*self.bound_layers, self.bound_layers)
        init.uniform(self.sm.bias, -1*self.bound_layers, self.bound_layers)


    def forward(self, inputs):
        """

        :param inputs: list of lists of form [left_tree, right_tree], to support minibatch of size > 1
        :return: outputs, tensor of dimensions batch_size x num_classes
        """

        # handles multiple inputs, inserted as list
        outputs = Variable(torch.rand(len(inputs), self.num_rels))

        for idx, input in enumerate(inputs):
            left = input[0]
            right = input[1]
            left_cps = self.compose(left)
            right_cps = self.compose(right)
            apply_cpr = self.cpr(torch.cat((left_cps, right_cps)))

            # compute kronecker product for child nodes, multiply this with cps tensor
            kron = torch.ger(left_cps,right_cps).view(-1)
            apply_cpr_t = self.cpr_t(kron)
            activated_cpr = self.relu(apply_cpr + apply_cpr_t)
            to_softmax = self.sm(activated_cpr).view(1, self.num_rels)
            output = F.softmax(to_softmax)
            outputs[idx,:] = output

        return(outputs) # size batch_size x num_classes


    def compose(self, tree):
        if tree.label() == '.': # leaf nodes
            word_onehot = self.word_dict[tree[0]]
            return self.voc(word_onehot) # get word embedding
        else:
            cps = self.cps(torch.cat((self.compose(tree[0]), self.compose(tree[1]))))

            # compute kronecker product for child nodes
            kron = torch.ger(self.compose(tree[0]), self.compose(tree[0])).view(-1)
            cps_t = self.cps_t(kron)
            activated_cps = self.tanh(cps + cps_t)
            return activated_cps

