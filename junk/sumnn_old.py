from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class sumNN_old(nn.Module):
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

        # a summing NN baseline which is largely identical to the TreeRNN, except that instead of
        # using a learned composition function, it simply sums the term vectors in each expression to compose
        # them before passing them to the comparison layer

        # composition matrix
        # self.cps = nn.Linear(2 * self.word_dim, self.word_dim)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.word_dim, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {word: i for i, word in enumerate(vocab)}

        # self.word_dict = {}
        # for word in vocab:
        #     # create one-hot encodings for words in vocabulary
        #     # self.word_dict[word] = Variable(torch.eye(self.voc_size)[:,vocab.index(word)], requires_grad=True)
        #     self.word_dict[word] = Variable(torch.LongTensor([vocab.index(word)])) #.view(-1)

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

            #nonlinearity only applied here
            left_cps = self.tanh(left_cps)
            right_cps = self.tanh(right_cps)

            # sum or concatenate sentence vectors?
            concat = torch.cat((left_cps, right_cps), 1)
            apply_cpr = self.cpr(concat)
            activated_cpr = self.relu(apply_cpr)
            to_softmax = self.sm(activated_cpr).view(1, self.num_rels)
            # NLL loss function requires log probabilities! so must use log_softmax here instead of softmax:
            output = F.log_softmax(to_softmax)
            outputs[idx,:] = output
        return(outputs) # size: batch_size x num_classes

    def compose(self, tree):

        if tree.label() == '.': # leaf nodes: get word embedding
            idx = [self.word_dict[tree[0]]]
            tensor = torch.LongTensor(idx)
            tensor_var = Variable(tensor)
            embedded = self.voc(tensor_var)


            #embedded = self.voc(self.word_dict[tree[0]]).view(-1)
            return(embedded)

        else:
            summed = self.compose(tree[0]) + self.compose(tree[1])
            #cps = self.cps(concat)
            #activated_cps = self.tanh(cps)
            # nonlinearity?
            #summed = self.tanh(summed) # this should not have been here
            return summed

if True:
    import datamanager as dat
    import torch.optim as optim
    from test import compute_accuracy

    sequential_loading = False
    word_dim = 25
    cpr_dim = 75
    bound_layers = None
    bound_embeddings = None
    l2_penalty = 1e-3
    batch_size = 32
    shuffle_samples = True
    num_epochs = 50
    test_all_epochs = True

    #train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/junk/nl_short_sentencestrain.txt'
    train_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
    train_data = dat.SentencePairsDataset(train_data_file)
    train_data.load_data(sequential=sequential_loading, print_result=True)

    batches = dat.BatchData(train_data, batch_size, shuffle_samples)
    batches.create_batches()

    vocab = train_data.word_list
    rels = train_data.relation_list

    #test_data = train_data

    test_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=sequential_loading)

    sumnn = sumNN_old(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim, bound_layers=bound_layers, bound_embeddings=bound_embeddings)

    criterion = nn.NLLLoss()

    optimizer = optim.Adadelta(sumnn.parameters(), weight_decay=l2_penalty)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # print('EPOCH ', str(epoch + 1))
        running_loss = 0.0

        # shuffle at each epoch
        if shuffle_samples and epoch > 0:
            batches = dat.BatchData(train_data, batch_size, shuffle_samples)
            batches.create_batches()

        for i in range(batches.num_batches):

            inputs = batches.batched_data[i]
            labels = batches.batched_labels[i]

            # convert label symbols to tensors
            labels = [rels.index(label) for label in labels]

            targets = Variable(torch.LongTensor(labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = sumnn(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print loss statistics
            # if show_loss:
            #     running_loss += loss.data[0]
            #     if (i + 2) % 100 == 1:  # print every 100 mini-batches
            #         print('[%d, %5d] loss: %.3f' %
            #               (epoch + 1, i, running_loss / 100))
            #         running_loss = 0.0

                    # bar.update(i + 1)

        if test_all_epochs and epoch < (num_epochs - 1):
            acc = compute_accuracy(test_data, rels, sumnn, print_outputs=False)
            print(str(epoch + 1), '\t', str(acc))