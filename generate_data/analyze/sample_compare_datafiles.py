from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def compare_files(nl_file, fol_file):
    conflicts_counter = Counter()

    with open(fol_file, 'r') as fol_f:
        for idxfol, line in enumerate(fol_f):
            if idxfol % 100 == 0:
                print('Sentence ', idxfol)
            all = line.split('\t')
            #print(all)
            left_fol = all[1]
            right_fol = all[2]
            rel_fol = all[0]
            with open(nl_file, 'r') as nl_f:
                for idxnl, line in enumerate(nl_f):
                    #print('nl')
                    #print(idxnl)
                    if idxnl >= idxfol:
                        all = line.split('\t')
                        #print(all)
                        left_nl = all[1]
                        right_nl = all[2]
                        rel_nl = all[0]
                        if left_fol == left_nl and right_fol == right_nl:
                        #print('same')
                            if rel_nl != rel_fol:
                                conflicts_counter[rel_nl + rel_fol] += 1
                                print(left_nl)
                                print(right_nl)
                                print('nl:')
                                print(rel_nl)
                                print('fol:')
                                print(rel_fol)
            if idxfol == 500:
                break
    return(conflicts_counter)

nl_train = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/final/nl/nl_data1_animals_train.txt'
fol_train = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/fol_animals_train_translated_from_nl.txt'
#compare_files(nl_train, fol_train)

def visualize(counter):
    """

    :param counters: list of Counter() objects
    :return:
    """

    #cumulative_counts = sum(counters, Counter())

    # average wrt number of samples
    #for key in cumulative_counts:
    #    cumulative_counts[key] /= SAMPLES

    # plot

    #labels, values = zip(*cumulative_counts.items())
    labels, values = zip(*counter.items())
    sorted_values = sorted(values)[::-1]
    sorted_labels = [x for (y,x) in sorted(zip(values,labels))][::-1]
    indexes = np.arange(len(sorted_labels))
    width = 1

    plt.bar(indexes, sorted_values)
    plt.xticks(indexes + width * 0.5, sorted_labels)

    plt.title("Frequencies of NL vs FOL conflicts in train data (total 22k instances)")
    plt.xlabel("Conflict")
    plt.ylabel("Frequency")
    plt.savefig("conflicts_fol_translation_of_nl_f1")
    #plt.show()
    plt.close()

# c = Counter({'v#': 2356, '|#': 2020, '>#': 1425, '<#': 1389, '<>': 638, '><': 632, '|v': 517, 'v|': 401, '|<': 136, 'v>': 136, '|^': 75, 'v^': 75, '#<': 24, '#>': 24, '#|': 22, '#v': 22})
# print(sum(c.values()))

#print(989200/21856)

total = 0
ind_count = 0
with open(nl_train, 'r') as nl_f:
    for idx, line in enumerate(nl_f):
        all = line.split('\t')
        #print(all)
        rel_fol = all[0]
        if rel_fol == '#':
            ind_count += 1
        total += 1
print('total:')
print(total)
print('independent:')
print(ind_count)

print(100*(ind_count/total))

