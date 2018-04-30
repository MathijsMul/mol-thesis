"""
organize data for hierarchic generalization experiment
"""

from collections import Counter

DET_SUBJ_LIST = ['some', 'all', 'not_some', 'not_all']

def analyze_file(data_file):
    rels = ['=', '<', '>', 'v', '^', '|', '#']
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
        rel_freq = round(100 * freq_dict[rel] / total, 2)
        rel_freq_dict[rel] = str(rel_freq)

    def dic_to_tex(dic):
        rels_to_tex = {'#': '\#', '<': '<', '=': '=', '>': '>', '^': '\wedge', 'v': '\lor', '|': '\mid'}
        s = '$\{'
        for key in dic.keys():
            s += rels_to_tex[key] + ' : ' + dic[key] + '\%, '
        s += '\}$'
        return(s)

    return(rel_freq_dict, total)

def get_det_subject_sentence(sentence):
    if sentence[2] == 'some':
        return('some')
    elif sentence[2] == 'all':
        return('all')
    elif sentence[3] == 'not':
        if sentence[4] == 'some':
            return('not_some')
        elif sentence[4] == 'all':
            return('not_all')

def get_det_subject_pair(line):
    line_tabs = line.split('\t')
    left = line_tabs[1].split()
    right = line_tabs[2].split()
    det_subj_left = get_det_subject_sentence(left)
    det_subj_right = get_det_subject_sentence(right)
    return(det_subj_left, det_subj_right)

def split_data_file(file):
    subfiles = {}
    subfile_instances = {}

    bare_filename = file.split('/')[-1][:-4]
    dir = '/'.join(file.split('/')[:-1]) + '/hierarchic_gen/'

    for item1 in DET_SUBJ_LIST:
        for item2 in DET_SUBJ_LIST:
            filename = dir + bare_filename + '_' + item1 + item2 + '.txt'
            subfiles[(item1, item2)] = filename
            subfile_instances[(item1, item2)] = []

    with open(file, 'r') as fin:
        for idx, line in enumerate(fin):
            dets_subj = get_det_subject_pair(line)
            subfile_instances[dets_subj] += [line]

    subfiles_list_shell = ''
    for subfile in subfiles.items():
        with open(subfile[1], 'w') as f:
            for item in subfile_instances[subfile[0]]:
                f.write(item)
        subfiles_list_shell += subfile[1] + ' '