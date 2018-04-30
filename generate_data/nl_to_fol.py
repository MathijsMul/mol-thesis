"""
translate NL data to FOL labels
"""

import fol_gen as fg

def extend_sentence(split_sentence):
    print(split_sentence)
    if split_sentence[3] != '(':
        split_sentence = split_sentence[:3] + ['(', ''] + [split_sentence[3]] + [')'] + split_sentence[4:]
    if split_sentence[8] != '(':
        split_sentence = split_sentence[:8] + ['(', ''] + [split_sentence[8]] + [')'] + split_sentence[9:]
    return(split_sentence)

def translate(nl_file, fol_file):
    with open(nl_file, 'r') as nl_f:
        with open(fol_file, 'w') as fol_f:
            for idx, line in enumerate(nl_f):
                all = line.split('\t')
                left = all[1]
                right = all[2]

                l = extend_sentence(left.split())
                #premise = [l[2], l[4], l[5], l[9], l[10]]
                r = extend_sentence(right.split())
                #hypothesis = [r[2], r[4], r[5], r[9], r[10]]

                if not ('most' in left + right or 'not_most' in left + right):
                    s = [[(l[2], r[2]), [(l[4], r[4]), (l[5], r[5])]], [(l[9], r[9]), (l[10], r[10])]]
                    #s = [[[(l[2], r[2]), (l[4], r[4])], [(na_subj1, na_subj2), (n_subj1, n_subj2)]], [[(va_subj1, va_subj2), (v_subj1, v_subj2)], [[(da_obj1, da_obj2), (det_obj1, det_obj2)], [(va_obj1, va_obj2), (n_obj1, n_obj2)]]]]

                    filtered_axioms = fg.filter_axioms(fg.axioms, l[2], r[2], l[4], r[4], l[5], r[5], l[10], r[10])
                    rel = fg.interpret(s, filtered_axioms)

                    train_line = rel + '\t' + left + '\t' + right
                    fol_f.write(train_line)

                    if idx % 100 == 0:
                        print('Translating sentence ', idx)