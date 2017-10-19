#data_file = 'junk/arithmdat.txt'
#data_file = 'data/final/fol/fol_data1_animals_train.txt'
data_file = 'generate_data/binary1_train.txt'

rels = ['=', '<', '>', 'v', '^', '|', '#']
# rels = [0,1,2,3,4]
# rels = ['0', '1', '2', '3', '4']
freq_dict = {}

for rel in rels:
    freq_dict[rel] = 0

total = 0

with open(data_file, 'r') as f:
    for idx, line in enumerate(f):
        all = line.split('\t')
        label = all[0]
        freq_dict[label] += 1
        total += 1

rel_freq_dict = {}
for rel in rels:
    rel_freq_dict[rel] = freq_dict[rel] / total

print(freq_dict)
print(rel_freq_dict)