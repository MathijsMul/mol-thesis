# fix problem with negations in data file: no 'not' included for objects, just brackets around single terms

def fix_line(line):
    split = line.split('\t')

    if '' in split:
        split.remove('')

    new_line = split[0]

    sentences = [split[1], split[2]]

    for s in sentences:
        s = s.split(' ')

        if '' in s:
            for i in s:
                if i == '':
                    s.remove(i)

        if s[-1] == '\n':
            del s[-1]

        if s[-4:] == [')', ')', ')', ')']:
            s.insert(-5, 'not')

        s = ' '.join(s)
        new_line += '\t' + s
    return(new_line)

def fix_file(file_in):
    with open(file_in, 'r') as fin:
        file_out = file_in + 'neg_corrected.txt'
        with open(file_out, 'w') as fout:
            for idx, line in enumerate(fin):
                new_line = fix_line(line)
                fout.write(new_line)
                fout.write('\n')

# f = '/Users/Mathijs/Documents/School/MoL/thesis/thesis_code/data/binary/bulk_binary_4negs.txt'
# fix_file(f)

# s = '#	( ( all Romans ) ( ( not hate ) ( all ( not Romans ) ) ) )	( ( all Germans ) ( ( not fear ) ( all ( Germans ) ) ) )'
# print(fix_line(s))

f = 'bulk_binary_4negs_uncorrected.txt'
fix_file(f)
#
# with open(f, 'r') as fin:
#     for idx, line in enumerate(fin):
#         print(fix_line(line))
#         if idx == 15:
#             break