import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import math

# input (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.
# h_0 (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
# Outputs: output, h_n
# output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_k) from the last layer of the RNN, for each k. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
# h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for k=seq_len.

rnn = nn.RNN(10, 20, 2)
input = Variable(torch.randn(5, 3, 10)) #(seq_len, batch, input_size)
h0 = Variable(torch.randn(2, 3, 20)) # (num_layers * num_directions, batch, hidden_size)
output, hn = rnn(input, h0)
#print(output) # (seq_len, batch, hidden_size * num_directions) 5x3x20

z = output.view(3, 5, 20)

#print(z)

a = Variable(torch.randn(2,4,6))
print(a)
output = a[1, ::].view(24)  # (num_layers * num_directions, batch, hidden_size)
print(output)
