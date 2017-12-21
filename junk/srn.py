import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math

#TODO: COMBINE SRN, GRU AND LSTM INTO ONE CLASS

class SRN(nn.Module):
    # Simple Recurrent Network (Elman)

    def __init__(self, vocab, rels, word_dim, hidden_size, cpr_dim):
        super(SRN, self).__init__()

        self.word_dim = word_dim  # dimensionality of word embeddings
        self.cpr_dim = cpr_dim  # output dimensionality of comparison layer
        self.rels = rels
        self.num_rels = len(rels)  # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        self.hidden_size = hidden_size

        # Simple Recurrent Network is RNN with single hidden layer
        self.simple_rnn = nn.RNN(self.word_dim, self.hidden_size, 1)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.hidden_size, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {word: i for i, word in enumerate(vocab)}

        # activation functions
        self.relu = nn.LeakyReLU()

        self.log_softmax = nn.LogSoftmax()

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


    def make_sentence_vector(self, sentence, pad_to_length):
        idxs = [self.word_dict[word] for word in sentence]
        tensor = torch.LongTensor(idxs)
        tensor_var = Variable(tensor)
        embedding = self.voc(tensor_var)

        length = len(sentence)

        if length < pad_to_length:
            pad = -1 * Variable(torch.ones(pad_to_length - length, self.word_dim))
            embedding = torch.cat((embedding, pad), 0)

        return(embedding)

    def make_batch_matrix(self, batch, size_batch):
        seq_lengths = [len(sequence) for sequence in batch]  # list of integers holding information about the batch size at each sequence step
        max_length = seq_lengths[0]

        # make container
        batch_out = Variable(torch.zeros((max_length, size_batch, self.word_dim)))

        for i in range(size_batch):
            padded_sentence_vector = self.make_sentence_vector(batch[i], pad_to_length=max_length)
            batch_out[:,i] = padded_sentence_vector

        # pack data
        pack = torch.nn.utils.rnn.pack_padded_sequence(batch_out, seq_lengths, batch_first=False)

        return(pack)

    def sort_inputs(self, inputs):
        inputs_ordered = sorted(enumerate(inputs), key=lambda k: len(k[1]), reverse=True)
        order = [item[0] for item in inputs_ordered]
        inputs_ordered = [item[1] for item in inputs_ordered]

        return(inputs_ordered, order)

    def forward(self, inputs):
        size_batch = len(inputs)

        left_inputs = [input[0] for input in inputs]
        right_inputs = [input[1] for input in inputs]

        left_rnn = self.rnn_forward(left_inputs, size_batch)
        right_rnn = self.rnn_forward(right_inputs, size_batch)

        concat = torch.cat((left_rnn, right_rnn), 1)
        apply_cpr = self.cpr(concat)
        activated_cpr = self.relu(apply_cpr)
        to_softmax = self.sm(activated_cpr)
        outputs = self.log_softmax(to_softmax)  # take log for NLLLoss

        return(outputs)  # size: batch_size x num_classes

    def rnn_forward(self, input, size_batch):
        input_sorted, order = self.sort_inputs(input)
        batch_matrix = self.make_batch_matrix(input_sorted, size_batch)

        hidden = Variable(torch.zeros(1, size_batch, self.hidden_size))
        output, _ = self.simple_rnn(batch_matrix, hidden) # output dim: (seq_len, batch, hidden_size * num_directions)

        # unpack
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output) # more than twice as fast
        outputs = [unpacked[unpacked_len[i]-1,::][i,:] for i in range(size_batch)]

        # restore original order
        old_order = sorted([(order[idx], outputs[idx]) for idx in range(size_batch)], key=lambda k : k[0])
        outputs_reordered = torch.stack([item[1] for item in old_order])

        return outputs_reordered