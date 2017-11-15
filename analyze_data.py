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

    return(total, freq_dict, rel_freq_dict)

# print(analyze_file('/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/negate_det1/split/binary1_neg_det1_train.txt'))
print(analyze_file('/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/negate_det1/split/binary1_neg_det1_train.txt'))

#f = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/generate_data/binary1_4negstrain.txt'
#f = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/negate_noun1/split/binary1_neg_noun1_test.txt'
#f = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/neg_det1_noun1_verb_noun2/binary2_4negs_train.txt'
f = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/generate_data/binary_2dets_4negs_train.txt'
print(analyze_file(f))
t = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/generate_data/binary_2dets_4negs_test.txt'
print(analyze_file(t))