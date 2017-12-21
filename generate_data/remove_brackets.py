import random

def adapt_sentence(sentence, ratio):
    s = sentence.split()

    t = []
    for word in s:
        if word in ['(', ')']:
            if random.random() < ratio:
                t += [word]
        else:
            t += [word]

    return(t)

# TODO: remove single brackets (as is done now), or only pairs?

def remove_brackets(file, ratio):
    file_out = file[:-4] + '_' + str(ratio) + 'brackets.txt'
    with open(file, 'r') as fin:
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                rel, s1, s2 = line.split('\t')
                s1, s2 = adapt_sentence(s1, ratio), adapt_sentence(s2, ratio)
                new_line = rel + '\t' + ' '.join(s1) + '\t' + ' '.join(s2) + '\n'
                fout.write(new_line)

# f = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
# for ratio in [i / 10.0 for i in range(11)]:
#     remove_brackets(f, ratio)

def get_bracket_pairs(sentence):
    stack = []
    bracket_pairs = []

    for idx, word in enumerate(sentence):
        if word == '(':
             stack.append(idx)
        if word == ')':
            try:
                #d[stack.pop()] = idx
                bracket_pairs += [(stack.pop(), idx)]
            except IndexError:
                print('Too many closing parentheses')
    if stack:  # check if stack is empty afterwards
        print('Too many opening parentheses')
    return(bracket_pairs)

def remove_bracket_pairs_sentence(sentence, ratio):
    sentence = sentence.split()
    indices_brackets = get_bracket_pairs(sentence)
    for pair in indices_brackets:
        if random.random() > ratio:
            sentence[pair[0]] = ''
            sentence[pair[1]] = ''

    new_sentence = [word for word in sentence if word != '']
    return(new_sentence)

def remove_bracket_pairs(file, ratio):
    file_out = file[:-4] + '_' + str(ratio) + 'bracket_pairs.txt'
    with open(file, 'r') as fin:
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                rel, s1, s2 = line.split('\t')
                s1, s2 = remove_bracket_pairs_sentence(s1, ratio), remove_bracket_pairs_sentence(s2, ratio)
                new_line = rel + '\t' + ' '.join(s1) + '\t' + ' '.join(s2) + '\n'
                fout.write(new_line)

#remove_bracket_pairs('/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_train.txt', 0.5)
f = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/binary_2dets_4negs_test.txt'
remove_bracket_pairs(f, 0.5)