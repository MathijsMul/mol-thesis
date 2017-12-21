import torch
from rnn import RNN
import datamanager as dat

# load data
data_file = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train.txt'
data = dat.SentencePairsDataset(data_file)
data.load_data(sequential=True)

# load model
vocab =   ['(', ')', 'Europeans', 'Germans', 'Italians', 'Romans', 'all', 'children', 'fear', 'hate', 'like', 'love', 'not', 'some']
rels = ['#', '<', '=', '>', '^', 'v', '|']

word_dim = 25
n_hidden = 128
cpr_dim = 75

model_path = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/models/rnns/gru/GRUbinary_2dets_4negs_train_ada_nodrop_3.pt'
net = RNN('GRU', vocab, rels, word_dim, n_hidden, cpr_dim)
net.load_state_dict(torch.load(model_path))

net.eval()

shuffle_samples = False
batch_size = 50
batches = dat.BatchData(data, batch_size, shuffle_samples)
batches.create_batches()

for batch_idx in range(batches.num_batches):
    print('Batch %i / %i' % (batch_idx, batches.num_batches))
    inputs = batches.batched_data[batch_idx]
    net(inputs)
