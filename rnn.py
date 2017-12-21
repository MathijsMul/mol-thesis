import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math
from diagnostic_hypothesis import diagnostic_labels

class RNN(nn.Module):
    # Simple Recurrent Network (Elman)

    def __init__(self, rnn_type, vocab, rels, word_dim, hidden_size, cpr_dim, p_dropout=0, layers=1):
        super(RNN, self).__init__()

        self.word_dim = word_dim  # dimensionality of word embeddings
        self.cpr_dim = cpr_dim  # output dimensionality of comparison layer
        self.rels = rels
        self.num_rels = len(rels)  # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.layers = layers

        if self.rnn_type == 'SRN':
            # Simple Recurrent Network is RNN with single hidden layer
            self.rnn = nn.RNN(self.word_dim, self.hidden_size, self.layers)
        elif self.rnn_type in ['GRU', 'GRU_connected']:
            # GRU with single hidden layer
            self.rnn = nn.GRU(self.word_dim, self.hidden_size, self.layers)
        elif self.rnn_type == 'LSTM':
            # LSTM with single hidden layer
            self.rnn = nn.LSTM(self.word_dim, self.hidden_size, self.layers)

        # comparison matrix
        self.cpr = nn.Linear(2 * self.hidden_size, self.cpr_dim)

        # matrix to softmax layer
        self.sm = nn.Linear(self.cpr_dim, self.num_rels)

        self.word_dict = {word: i for i, word in enumerate(vocab)}

        # activation functions
        self.relu = nn.LeakyReLU()

        # dropout layer
        self.dropout = nn.Dropout(p_dropout)

        self.log_softmax = nn.LogSoftmax()

        self.diagnostic = True

        if self.diagnostic:
            diagnostic_file = 'delete.txt'
            self.diagnostic_data = open(diagnostic_file, 'w')


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

        restore_order = [idx for idx, new_idx in sorted(enumerate(order), key=lambda k: k[1])]

        return(inputs_ordered, order, restore_order)

    def forward(self, inputs):
        #print(inputs)
        size_batch = len(inputs)

        left_inputs = [input[0] for input in inputs]
        right_inputs = [input[1] for input in inputs]

        left_rnn = self.rnn_forward(left_inputs, size_batch)

        if self.rnn_type == 'GRU_connected':
            right_rnn, _ = self.rnn_forward(right_inputs, size_batch, left_rnn)
        else:
            right_rnn, _ = self.rnn_forward(right_inputs, size_batch)

        concat = torch.cat((left_rnn, right_rnn), 1)
        apply_cpr = self.cpr(concat)
        activated_cpr = self.relu(apply_cpr)

        # apply dropout here: close to output, after non-linearity, before parameterized linear transformation,
        # deactivated during evaluation
        apply_dropout = self.dropout(activated_cpr)

        to_softmax = self.sm(apply_dropout)
        outputs = self.log_softmax(to_softmax)  # take log for NLLLoss

        return(outputs)  # size: batch_size x num_classes

    def rnn_forward(self, input, size_batch, hypothesis=None, init_hidden=None):
        input_sorted, order, restore_order = self.sort_inputs(input)
        batch_matrix = self.make_batch_matrix(input_sorted, size_batch)

        if not init_hidden is None:
            # order hidden state in same order as input
            order_tensor = torch.LongTensor(order)
            init_hidden = init_hidden[order_tensor, ::]
            # reshape to (1xbatchxhidden_size)
            init_hidden = init_hidden.unsqueeze(0)
            output, _ = self.rnn(batch_matrix, init_hidden)
        else:
            # hidden = Variable(torch.zeros(1, size_batch, self.hidden_size)) defaults to zeros anyway
            output, _ = self.rnn(batch_matrix)  # output dim: (seq_len, batch, hidden_size * num_directions)

        # unpack
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
        outputs = [unpacked[unpacked_len[i]-1,::][i,:] for i in range(size_batch)]

        # restore original order
        outputs_reordered = torch.stack([outputs[restore_order[i]] for i in range(size_batch)])
        # old_order = sorted([(order[idx], outputs[idx]) for idx in range(size_batch)], key=lambda k : k[0])
        # outputs_reordered = torch.stack([item[1] for item in old_order])

        if self.diagnostic:
            # get labels
            labels = diagnostic_labels(input, hypothesis)

            # restore original order
            restore_order_tensor = torch.LongTensor(restore_order)
            #print(restore_order)
            unpacked_reordered = unpacked[:,restore_order_tensor, :]
            #print(unpacked_reordered.size())
            lengths_reordered = [unpacked_len[restore_order[i]] for i in range(size_batch)]
            #print(lengths_reordered)

            hidden_vectors = []
            for item_idx in range(size_batch):
                hidden_vectors_sentence = []
                for seq_idx in range(lengths_reordered[item_idx]):
                    hidden_vector = unpacked_reordered[seq_idx,:][item_idx,:].data
                    hidden_vector = hidden_vector.view(1, self.hidden_size).tolist()[0]
                    #print(hidden_vector)
                    hidden_vectors_sentence += [hidden_vector]
                    label = labels[item_idx][seq_idx]
                    diagnostic_data_line = str(label) + '\t' + str(hidden_vector) + '\n'
                    self.diagnostic_data.write(diagnostic_data_line)
                hidden_vectors += [hidden_vectors_sentence]
        else:
            hidden_vectors = None

            # outputs = [unpacked[unpacked_len[i] - 1, ::][i, :] for i in range(size_batch)]
            # outputs_reordered = [outputs[order[i]] for i in range(len(outputs))]

        return outputs_reordered, hidden_vectors

    # def diagnostic_labels(self, input, hypothesis):
    #     labels = []
    #
    #     if hypothesis == 'brackets':
    #         for sentence in input:
    #             sentence_labels = []
    #             depth = 0
    #             for word in sentence:
    #                 if word == '(':
    #                     depth += 1
    #                 elif word == ')':
    #                     depth -= 1
    #                 sentence_labels += [depth]
    #             labels += [sentence_labels]
    #
    #     elif hypothesis == 'length':
    #         for sentence in input:
    #             sentence_labels = []
    #             length = 0
    #             for word in sentence:
    #                 length += 1
    #                 sentence_labels += [length]
    #             labels += [sentence_labels]
    #
    #     elif hypothesis == 'pos':
    #         pos_tags = ['bracket', 'noun', 'verb', 'quant', 'neg']
    #         pos_mapping = {pos_tag : idx for idx, pos_tag in enumerate(pos_tags)}
    #
    #         for sentence in input:
    #             sentence_labels = []
    #             for word in sentence:
    #                 if word in ['(', ')']:
    #                     sentence_labels += [pos_mapping['bracket']]
    #                 elif word in ['Europeans', 'Germans', 'Italians', 'Romans', 'children']:
    #                     sentence_labels += [pos_mapping['noun']]
    #                 elif word in ['fear', 'hate', 'like', 'love']:
    #                     sentence_labels += [pos_mapping['verb']]
    #                 elif word in ['all', 'some']:
    #                     sentence_labels += [pos_mapping['quant']]
    #                 elif word in ['not']:
    #                     sentence_labels += [pos_mapping['neg']]
    #             labels += [sentence_labels]
    #
    #     return(labels)

if False:
    import datamanager as dat
    sequential_loading = True
    word_dim = 25
    hidden_size = 128
    cpr_dim = 75
    num_epochs = 10
    from test import compute_accuracy

    #train_data_file = 'data/junk/nl_short_sentencestrain.txt'
    train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
    train_data = dat.SentencePairsDataset(train_data_file)
    train_data.load_data(sequential=sequential_loading, print_result=True)


    vocab = train_data.word_list
    rels = train_data.relation_list
    test_data = train_data
    batch_size = 32
    shuffle_samples = False

    batches = dat.BatchData(train_data, batch_size, shuffle_samples)
    batches.create_batches()

    gru = RNN('GRU', vocab,rels, word_dim,hidden_size,cpr_dim,p_dropout=0)

    for i in range(batches.num_batches):
        inputs = batches.batched_data[i]
        outputs = gru(inputs)

    exit()

    criterion = nn.NLLLoss()

    optimizer = optim.Adadelta(gru_conn.parameters())

    #acc_before_training = compute_accuracy(test_data, rels, gru_conn, print_outputs=False, confusion_matrix=False)
    #print("EPOCH", "\t", "ACCURACY")
    #print(str(0), '\t', str(acc_before_training))

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # print('EPOCH ', str(epoch + 1))
        running_loss = 0.0

        # shuffle at each epoch
        if epoch > 0:
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
            outputs = gru_conn(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch < (num_epochs - 1):
            acc = compute_accuracy(test_data, rels, gru_conn, print_outputs=False)
            print(str(epoch + 1), '\t', str(acc))