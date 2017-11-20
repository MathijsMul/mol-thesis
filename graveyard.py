from __future__ import print_function
import torch


n_lengths = 3
totals = torch.zeros(n_lengths, n_lengths)
errors = torch.eye(n_lengths, n_lengths)

n = totals /errors
print(n)