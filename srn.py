import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math

class SRN(nn.Module):
    # Simple Recurrent Network

    def __init__(self, vocab, rels, word_dim, hidden_size, cpr_dim):
    #def __init__(self, input_size, hidden_size, output_size):
        super(SRN, self).__init__()

        self.word_dim = word_dim  # dimensionality of word embeddings
        self.cpr_dim = cpr_dim  # output dimensionality of comparison layer
        self.num_rels = len(rels)  # number of relations (labels)
        self.voc_size = len(vocab)

        self.voc = nn.Embedding(self.voc_size, self.word_dim)

        #self.input_size = len(vocab) : became self.voc_size
        self.hidden_size = hidden_size

        #self.output_size = len(rels): became self.num_rels

        # self.i2h = nn.Linear(self.word_dim + self.hidden_size, self.hidden_size)
        # self.i2o = nn.Linear(self.word_dim + self.hidden_size, self.hidden_size)

        # Simple Recurrent Network is RNN with single hidden layer
        self.simple_rnn = nn.RNN(self.word_dim, self.hidden_size, 1)

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
            pass
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

        output, hn = self.simple_rnn(input_vector, hidden) # output dim: (num_layers * num_directions, batch, hidden_size)
        output = output[seq_len-1,::].view(1, self.hidden_size) # output dim: (1, hidden_size)
        return output

        #print(output)
        # for word in input:
        #     embedded = self.voc(self.word_dict[word]) #.view(-1)
            # combined = torch.cat((embedded, hidden), 1)
            # hidden = self.i2h(combined)
            # output = self.i2o(combined)
            # # output = self.softmax(output)

        #return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

if False:
    import datamanager as dat
    from test import compute_accuracy

    train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_train_translated_from_nl.txt_downsampled_0.01'
    train_data = dat.SentencePairsDataset(train_data_file)
    train_data.load_data(sequential=True, print_result=True)
    vocab = train_data.word_list
    rels = train_data.relation_list

    #test_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_test_translated_from_nl.txt_downsampled_0.01'
    test_file = train_data_file
    test_data = dat.SentencePairsDataset(test_file)
    test_data.load_data(sequential=True, print_result=False)

    batch_size = 32
    shuffle_samples = True

    batches = dat.BatchData(train_data, batch_size, shuffle_samples)
    batches.create_batches()

    word_dim = 25
    n_hidden = 128
    cpr_dim = 75

    #init_mode = 'xavier_uniform'
    rnn = SRN(vocab, rels, word_dim, n_hidden, cpr_dim)
    #
    # print(list(rnn.parameters()))
    #rnn.initialize(mode=init_mode)

    #
    #learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    criterion = nn.NLLLoss()

    l2_penalty = 1e-3
    optimizer = optim.Adadelta(rnn.parameters(), weight_decay = l2_penalty)

    num_epochs = 50
    for epoch in range(num_epochs):
        batches = dat.BatchData(train_data, batch_size, shuffle_samples)
        batches.create_batches()
        running_loss = 0.0
        for i in range(batches.num_batches):

            inputs = batches.batched_data[i]
            #print(inputs)
            labels = batches.batched_labels[i]

            # convert label symbols to tensors
            labels = [rels.index(label) for label in labels]
            targets = Variable(torch.LongTensor(labels))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = rnn(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print loss statistics
            #if show_loss:
            # running_loss += loss.data[0]
            # if i  % 100 == 1:  # print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 100, running_loss / 100))
            #     running_loss = 0.0

        acc = compute_accuracy(test_data, rels, rnn, print_outputs=False)
        print(str(epoch + 1), '\t', str(acc))