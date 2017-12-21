import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math

#TODO: CHECK IF NLLLOSS CRITERION RECEIVES OUTPUTS AND TARGETS IN RIGHT WAY

#TODO: ALSO UPDATE SRN AND LSTM

class GRU(nn.Module):
    # Gated Recurrent Unit

    def __init__(self, vocab, rels, word_dim, hidden_size, cpr_dim):
        super(GRU, self).__init__()

        self.word_dim = word_dim  # dimensionality of word embeddings
        self.cpr_dim = cpr_dim  # output dimensionality of comparison layer
        self.rels = rels
        self.num_rels = len(rels)  # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        self.hidden_size = hidden_size

        # GRU with single hidden layer
        self.gru = nn.GRU(self.word_dim, self.hidden_size, 1)

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
            # use torch function for padding?
            padded_sentence_vector = self.make_sentence_vector(batch[i], pad_to_length=max_length)
            batch_out[:,i] = padded_sentence_vector

        # pack data
        pack = torch.nn.utils.rnn.pack_padded_sequence(batch_out, seq_lengths, batch_first=False)
        return(pack)

        #Packs a Variable containing padded sequences of variable length.
        #Input can be of size TxBx* where T is the length of the longest sequence
              # (equal to lengths[0]), B is the batch size, and * is any number of dimensions (including 0).
        # If batch_first is True BxTx* inputs are expected.
        #The sequences should be sorted by length in a decreasing order, i.e. input[:,0] should be the longest sequence,
              # and input[:,B-1] the shortest one.



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

        #TODO
        # CHECK THIS
        to_softmax = self.sm(activated_cpr)

        # REPLACE BY OTHER SOFTMAX
        #outputs = F.log_softmax(to_softmax)

        outputs = self.log_softmax(to_softmax)  # take log for NLLLoss

        return(outputs)  # size: batch_size x num_classes

    def rnn_forward(self, input, size_batch):
        input_sorted, order = self.sort_inputs(input)
        batch_matrix = self.make_batch_matrix(input_sorted, size_batch)

        #hidden = Variable(torch.zeros(1, size_batch, self.hidden_size))

        output, _ = self.gru(batch_matrix) # hidden defaults to zeros anyway
        # output dim: (seq_len, batch, hidden_size * num_directions)

        # unpack
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output) # more than twice as fast

        #TODO: CHECK THIS with small dimension. is this really the output we need?
        outputs = [unpacked[unpacked_len[i]-1,::][i,:] for i in range(size_batch)]

        # restore original order
        old_order = sorted([(order[idx], outputs[idx]) for idx in range(size_batch)], key=lambda k : k[0])
        outputs_reordered = torch.stack([item[1] for item in old_order])

        return outputs_reordered



if False:
    import datamanager as dat
    import torch.optim as optim
    from test import compute_accuracy

    sequential_loading = True
    word_dim = 25
    cpr_dim = 75
    #bound_layers = 0.05
    #bound_embeddings = 0.01
    l2_penalty = 1e-3 #0.0003
    batch_size = 32
    shuffle_samples = True
    num_epochs = 50
    test_all_epochs = True
    hidden_size = 128

    #train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/junk/nl_short_sentencestrain.txt'
    #train_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
    #train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_train_translated_from_nl.txt'
    train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_train.txt'
    train_data = dat.SentencePairsDataset(train_data_file)
    train_data.load_data(sequential=sequential_loading, print_result=True)

    batches = dat.BatchData(train_data, batch_size, shuffle_samples)
    batches.create_batches()

    vocab = train_data.word_list
    rels = train_data.relation_list

    #test_data = train_data

    #test_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
    #test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_test_translated_from_nl.txt'
    test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_test.txt'
    test_data = dat.SentencePairsDataset(test_data_file)
    test_data.load_data(sequential=sequential_loading)

    net = GRU(vocab, rels, word_dim=word_dim, hidden_size=hidden_size, cpr_dim=cpr_dim)

    #TODO: if bad score, change to xavier
    #sumnn.initialize('uniform')
    #sumnn.initialize('xavier_uniform')

    criterion = nn.NLLLoss()

    optimizer = optim.Adadelta(net.parameters(), weight_decay=l2_penalty)

    acc_before_training = compute_accuracy(test_data, rels, net, print_outputs=False, confusion_matrix=False)
    print("EPOCH", "\t", "ACCURACY")
    print(str(0), '\t', str(acc_before_training))

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        #print('EPOCH ', str(epoch + 1))
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
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


        if test_all_epochs and epoch < (num_epochs - 1):
            acc = compute_accuracy(test_data, rels, net, print_outputs=False)
            print(str(epoch + 1), '\t', str(acc))


