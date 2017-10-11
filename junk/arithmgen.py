from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random

datafile = 'arithmdat.txt'

nums = [0, 1]
with open(datafile, 'w') as f:
    for line in range(500):

        dat = []
        total = 0
        for i in range(4):

            if random.random() > 0.5:
                dat += [nums[0]]
            else:
                dat += [nums[1]]
            total += dat[i]

        f.write(str(total) + '\t' + '( ' + str(dat[0]) + ' ' + str(dat[1]) + ' )\t( ' + str(dat[2]) + ' ' + str(dat[3]) + ' )')

        f.write('\n')


