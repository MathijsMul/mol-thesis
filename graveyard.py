from __future__ import print_function
import torch

# t = torch.LongTensor(5).random_(0, 100)
# s = torch.FloatTensor(5).random_(0, 100)
#
# print(t)
# print(s)
#
# x = torch.randn(2,5)
# print(x)
# x = x.long()
# print(x)

l = '#		 ( ( all Italians  ) ( ( not hate ) ( some Romans ) ) ) 	 ( ( all ( not Italians )  ) ( ( not like ) ( some Italians ) ) ) '

print(l)
print(l.replace(' \t\t ', '\t'))

split = l.split('\t')
print(split)