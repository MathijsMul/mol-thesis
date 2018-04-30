import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math
import torch.autograd as autograd

# input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.
# h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
# Outputs: output, h_n
# output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_k) from the last layer of the RNN, for each k. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
# h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for k=seq_len.

# rnn = nn.RNN(10, 20, 2)
# input = Variable(torch.randn(5, 3, 10)) #(seq_len, batch, input_size)
# h0 = Variable(torch.randn(2, 3, 20)) # (num_layers * num_directions, batch, hidden_size)
# output, hn = rnn(input, h0)
# #print(output) # (seq_len, batch, hidden_size * num_directions) 5x3x20
#
# z = output.view(3, 5, 20)
#
# #print(z)
#
# a = Variable(torch.randn(2,4,6))
# print(a)
# output = a[1, ::].view(24)  # (num_layers * num_directions, batch, hidden_size)
# print(output)

# t4d = torch.Tensor(3, 3, 3, 3)
# p1d = (0, 0, 0, 1) # pad last dim by 1 on each side
# out = F.pad(t4d, p1d, "constant", -1)
# # print(out)
# # print(out.data.size())
#
# input = Variable(torch.rand(7,25)) # 1D (N, C, L)
# length = input.data.size()[0]
# max_length = 10
# word_dim = 25
# pad = -1*torch.ones(max_length - length, word_dim)
# output = torch.cat((input, pad), 0)
# print(output)

# input_2d = input.unsqueeze(2) # add a fake height
# input_2d = input_2d.unsqueeze(2)
# p = F.pad(input_2d, (0, 2, 0, 0), value=-1).view(1, 1, -1) # left padding (and remove height)
# F.pad(input_2d, (0, 2, 0, 0)).view(1, 1, -1) # right padding (and remove height)
#
# print(input)
# print(input_2d)
# print(p)

#
# p = torch.Tensor(2,3, 4)
#
# # unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
# # outputs = [unpacked[unpacked_len[i],:i:] for i in range(size_batch)]
# print(p)
# o = p[1,::][0,:]
# print(o)
# # q = o[0,:]
# # print(q)


#
#  x = torch.randn(4)
# print(x)
# y, _ = torch.sort(x, 0)
# print(y)
# unsorted = y.new(*y.size())
# print(unsorted)
# ind2 = torch.LongTensor([3,2,1,0])
# unsorted.scatter_(0, ind2, y)
# print(unsorted)
#
# #s = [2, 3, 1, 4, 5]
# s = [[1,2,3], [1], [1,2], [1,2,3,4], [1,2], [1,2,3], [1,2]]
# print('original:')
# print(s)
# s_ordered = sorted(enumerate(s), key = lambda k : len(k[1]), reverse=True)
#
# #order = sorted(range(len(s)), key = lambda k : len(s_ordered[k]), reverse=True)
# order = [item[0] for item in s_ordered]
# ordered = [item[1] for item in s_ordered]
# print('order:')
# print(order)
# print('ordered:')
# print(ordered)
# print('restored:')
# restored = [item[1] for item in sorted([(order[idx], ordered[idx]) for idx in range(len(s))], key = lambda k : k[0], reverse=False)]
# print(restored)

# p = sorted(range(len(s)), key=lambda k: s[k])
# #[2, 0, 1, 3, 4]
# print(p)

# if False:
#     import datamanager as dat
#     import torch.optim as optim
#     from test import compute_accuracy
#
#     sequential_loading = True
#     word_dim = 25
#     cpr_dim = 75
#     bound_layers = 0.05
#     bound_embeddings = 0.01
#     l2_penalty = 0.0003
#     batch_size = 32
#     shuffle_samples = True
#     num_epochs = 50
#     test_all_epochs = True
#
#     #train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/junk/nl_short_sentencestrain.txt'
#     #train_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
#     #train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_train_translated_from_nl.txt'
#     train_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_train.txt'
#     train_data = dat.SentencePairsDataset(train_data_file)
#     train_data.load_data(sequential=sequential_loading, print_result=True)
#
#     batches = dat.BatchData(train_data, batch_size, shuffle_samples)
#     batches.create_batches()
#
#     vocab = train_data.word_list
#     rels = train_data.relation_list
#
#     #test_data = train_data
#
#     #test_data_file = './data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
#     #test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/fol/fol_animals_test_translated_from_nl.txt'
#     test_data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/unary/nl/nl_data1_animals_test.txt'
#     test_data = dat.SentencePairsDataset(test_data_file)
#     test_data.load_data(sequential=sequential_loading)
#
#     sumnn = sumNN(vocab, rels, word_dim=word_dim, cpr_dim=cpr_dim, bound_layers=bound_layers, bound_embeddings=bound_embeddings)
#
#     #TODO: if bad score, change to xavier
#     #sumnn.initialize('uniform')
#     sumnn.initialize('xavier_uniform')
#
#     criterion = nn.NLLLoss()
#
#     optimizer = optim.Adadelta(sumnn.parameters(), weight_decay=l2_penalty)
#
#     acc_before_training = compute_accuracy(test_data, rels, sumnn, print_outputs=False, confusion_matrix=False)
#     print("EPOCH", "\t", "ACCURACY")
#     print(str(0), '\t', str(acc_before_training))
#
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#
#         # print('EPOCH ', str(epoch + 1))
#         running_loss = 0.0
#
#         # shuffle at each epoch
#         if shuffle_samples and epoch > 0:
#             batches = dat.BatchData(train_data, batch_size, shuffle_samples)
#             batches.create_batches()
#
#         for i in range(batches.num_batches):
#
#             inputs = batches.batched_data[i]
#             labels = batches.batched_labels[i]
#
#             # convert label symbols to tensors
#             labels = [rels.index(label) for label in labels]
#
#             targets = Variable(torch.LongTensor(labels))
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = sumnn(inputs)
#
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             # print loss statistics
#             # if show_loss:
#             #     running_loss += loss.data[0]
#             #     if (i + 2) % 100 == 1:  # print every 100 mini-batches
#             #         print('[%d, %5d] loss: %.3f' %
#             #               (epoch + 1, i, running_loss / 100))
#             #         running_loss = 0.0
#
#                     # bar.update(i + 1)
#
#         if test_all_epochs and epoch < (num_epochs - 1):
#             acc = compute_accuracy(test_data, rels, sumnn, print_outputs=False)
#             print(str(epoch + 1), '\t', str(acc))

# max_length = 2
# size_batch = 3
# word_dim = 4
#
# batch_out = torch.randn((max_length, size_batch, word_dim))
# print(batch_out)
# print(batch_out[:,0])

# for i in range(size_batch):
#     padded_sentence_vector = self.make_sentence_vector(batch[i], pad_to_length=max_length)
#     batch_out[:,i] = padded_sentence_vector

# m = nn.Dropout(p=0.5)
# input = autograd.Variable(torch.randn(20, 16))

#n = np.arange(480).reshape((5,4,4,6))
# a = torch.randn((3,2,1))
# print(a)
# #a = torch.from_numpy(n)
# perm = torch.LongTensor([2,1,0])
# b = a[perm,: :]
# print(b)
#
# a = torch.randn((2,3,4))
# print(a)
# print(a[0,:][0,:])
#
# m = nn.LogSoftmax()
# loss = nn.NLLLoss()
# input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
# target = autograd.Variable(torch.LongTensor([1, 0, 4]))
# print(input)
# print(target)
# output = loss(m(input), target)
# print(output)


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math
from collections import defaultdict



import pandas as pd


df = pd.DataFrame(columns=['length1', 'length2'])
for i in range(10):
    df = df.append({'length1': i % 2, 'length2': i % 2}, ignore_index=True)

print(df)

pivoted_all = df.pivot(index='length1', columns='length2') # values='testing accuracy')
print(pivoted_all)
