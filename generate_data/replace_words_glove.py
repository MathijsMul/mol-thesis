"""
replace words in data file
"""

def replace(file, to_replace, replace_to):
    filename = ''.join(file.split('.')[-2])
    f_out_old_name = filename + to_replace + '.txt'
    f_out_old = open(f_out_old_name, 'w')
    f_out_new_name = filename + to_replace + '_to_' + replace_to + '.txt'
    f_out_new = open(f_out_new_name, 'w')

    with open(file, 'r') as f_in:
        for idx, line in enumerate(f_in):
            if to_replace in line:
                f_out_old.write(line)
                new_line = line.replace(to_replace, replace_to)
                f_out_new.write(new_line)

def replace_dict(file, replace_dict, old_name, new_name):
    filename = ''.join(file.split('.')[-2])
    f_out_old_name = filename + old_name + '.txt'
    f_out_old = open(f_out_old_name, 'w')
    f_out_new_name = filename + new_name + '.txt'
    f_out_new = open(f_out_new_name, 'w')

    with open(file, 'r') as f_in:
        for idx, line in enumerate(f_in):
            new_line = line
            for old_word in replace_dict:
                new_line = new_line.replace(old_word, replace_dict[old_word])
            if new_line != line:
                f_out_old.write(line)
                f_out_new.write(new_line)

f='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/data/binary/2dets_4negs/partial_bracketing/test/binary_2dets_4negs_test_0bracket_pairs.txt'

people_to_animals = {'Romans' : 'rabbits',
                     'Italians' : 'rodents',
                     'Germans' : 'cats',
                     'Europeans' : 'mammals',
                     'children' : 'pets'}

italy_to_france = {'Romans' : 'Parisians',
                   'Italians' : 'French'}

to_france_poland = {'Romans' : 'Parisians',
                   'Italians' : 'French',
                    'Germans' : 'Polish'}

to_france_poland_students = {'Romans' : 'Parisians',
                   'Italians' : 'French',
                    'Germans' : 'Polish',
                    'children' : 'students'}

people_to_religion = {'Romans' : 'calvinists',
                     'Italians' : 'protestants',
                     'Germans' : 'catholics',
                     'Europeans' : 'christians',
                     'children' : 'orthodox'}

people_to_america = {'Romans' : 'Clevelanders',
                     'Italians' : 'Ohioans',
                     'Germans' : 'Californians',
                     'Europeans' : 'Americans',
                     'children' : 'women'}

to_eurasians = {'Romans' : 'Parisians',
                   'Italians' : 'French',
                    'Germans' : 'Polish',
                    'children' : 'students',
                'Europeans' : 'Eurasians'}

fr_par_pol_eur = {'Romans' : 'Parisians',
                   'Italians' : 'French',
                    'Germans' : 'Polish',
                'Europeans' : 'Eurasians'}
