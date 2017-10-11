from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random

datafile = 'easydata.txt'

with open(datafile, 'w') as f:
    for line in range(500):
        data = [random.random() for i in range(5)]

        if sum(data) > 0.1:
            label = '1'
        else:
            label = '0'

        f.write(label + '\t')

        for d in data:
            f.write(str(d) + '\t')

        f.write('\n')


