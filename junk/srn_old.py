import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math

#TODO: optimize this, faster data loading/more lookups or sth

class SRN_old(nn.Module):
    # Simple Recurrent Network

    def __init__(self, vocab, rels, word_dim, hidden_size, cpr_dim):
    #def __init__(self, input_size, hidden_size, output_size):
        super(SRN_old, self).__init__()

        self.word_dim = word_dim  # dimensionality of word embeddings
        self.cpr_dim = cpr_dim  # output dimensionality of comparison layer
        self.num_rels = len(rels)  # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        #self.input_size = len(vocab) : became self.voc_size
        self.hidden_size = hidden_size

        # Simple Recurrent Network is RNN with single hidden layer
        self.simple_rnn = nn.RNN(self.word_dim, self.hidden_size, 1)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.hidden_size, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {word: i for i, word in enumerate(vocab)}

        # activation functions
        self.relu = nn.LeakyReLU()  # cpr layer, negative slope is 0.01, which is standard

    def initialize(self, mode):
        #Xavier uniform doesn't seem best choice for RNN

        # always initialize biases as zero vectors:
        # pass
        # self.cpr.bias.data.fill_(0)
        # self.sm.bias.data.fill_(0)
        # self.simple_rnn.bias_ih_l0.data.fill_(0)
        # self.simple_rnn.bias_hh_l0.data.fill_(0)

        if mode == 'xavier_uniform':
            raise(NotImplementedError)
            #pass
            # init.xavier_uniform(self.simple_rnn.weight_hh_l0, gain = 5/3)  # rec. gain for tanh
            # init.xavier_uniform(self.simple_rnn.weight_ih_l0, gain = 5/3)  # rec. gain for tanh
            # init.xavier_uniform(self.voc.weight)
            # init.xavier_uniform(self.cpr.weight, gain=math.sqrt(2 / (1 + (0.01 ** 2))))  # rec. gain for leakyrelu
            # init.xavier_uniform(self.sm.weight)

        if mode == 'xavier_normal':
            raise(NotImplementedError)
            # init.xavier_normal(self.voc.weight)
            # init.xavier_normal(self.cpr.weight, gain=math.sqrt(2 / (1 + (0.01 ** 2))))
            # init.xavier_normal(self.sm.weight)
            # init.xavier_normal(self.i2h.weight)
            # init.xavier_normal(self.i2o.weight)

        if mode == 'uniform':
            raise(NotImplementedError)
            # old
            # init.uniform(self.voc.weight, -1 * self.bound_embeddings, self.bound_embeddings)
            # init.uniform(self.cpr.weight, -1 * self.bound_layers, self.bound_layers)
            # init.uniform(self.sm.weight, -1 * self.bound_layers, self.bound_layers)
            # not implemented for i2h and i2o

    def make_sentence_vector(self, sentence):
        idxs = [self.word_dict[word] for word in sentence]
        #print(idxs)
        tensor = torch.LongTensor(idxs)
        return self.voc(Variable(tensor))

    def forward(self, inputs):
        # handles batch with multiple inputs, inserted as list
        outputs = Variable(torch.rand(len(inputs), self.num_rels))

        for idx, input in enumerate(inputs):
            left = input[0]
            #left_vector = self.make_sentence_vector(left)
            right = input[1]
            #right_vector = self.make_sentence_vector(right)

            left_rnn = self.rnn_forward(left)
            right_rnn = self.rnn_forward(right)

            concat = torch.cat((left_rnn, right_rnn), 1)

            # concatenate sentence/context vectors outputted by rnns
            apply_cpr = self.cpr(concat)

            activated_cpr = self.relu(apply_cpr)
            to_softmax = self.sm(activated_cpr).view(1, self.num_rels)

            # NLL loss function requires log probabilities! so must use log_softmax here instead of softmax:
            output = F.log_softmax(to_softmax)
            outputs[idx, :] = output

        return(outputs)  # size: batch_size x num_classes

    def rnn_forward(self, input):
        seq_len = len(input)
        input_vector = self.make_sentence_vector(input)
        input_vector = input_vector.view(seq_len, 1, self.word_dim)  # must be dim: (seq_len, batch, input_size)

        hidden = self.initHidden()

        output, hn = self.simple_rnn(input_vector, hidden) # output dim:  (seq_len, batch, hidden_size * num_directions)
        output = output[seq_len-1,::].view(1, self.hidden_size) # output dim: (1, hidden_size)

        output = self.dropout(output)
        return output

    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

        # maakt dit uit?
        #return Variable(torch.zeros(1, self.hidden_size))