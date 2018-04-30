"""
chunk large data files
"""

lines_per_file = 600000
smallfile = None

f = 'bulk2dets4negs_6bulk.txt'
with open(f) as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = 'chunked_{}_{}.txt'.format(f, lineno + lines_per_file)
            smallfile = open(small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()

f = 'bulk2dets4negs_4bulk.txt'
with open(f) as bigfile:
    for lineno, line in enumerate(bigfile):
        if lineno % lines_per_file == 0:
            if smallfile:
                smallfile.close()
            small_filename = 'chunked_{}_{}.txt'.format(f, lineno + lines_per_file)
            smallfile = open(small_filename, "w")
        smallfile.write(line)
    if smallfile:
        smallfile.close()

with open('bulk2dets4negs_1bulk.txt', 'r') as f:
    for idx, l in enumerate(f):
        continue
    print(idx)
