from collections import Counter

def analyze_file(data_file):
    rels = ['=', '<', '>', 'v', '^', '|', '#']
    # rels = [0,1,2,3,4]
    # rels = ['0', '1', '2', '3', '4']
    freq_dict = Counter()

    for rel in rels:
        freq_dict[rel] = 0

    total = 0

    with open(data_file, 'r') as f:
        for idx, line in enumerate(f):
            all = line.split('\t')
            label = all[0]
            freq_dict[label] += 1
            total += 1

    rel_freq_dict = Counter()
    for rel in rels:
        rel_freq_dict[rel] = freq_dict[rel] / total

    return(freq_dict, rel_freq_dict)

print(analyze_file('/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/negate_det1/split/binary1_neg_det1_train.txt'))
print(analyze_file('/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/negate_verb/split/binary1_neg_verbtrain.txt'))