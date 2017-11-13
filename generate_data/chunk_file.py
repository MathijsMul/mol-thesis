lines_per_file = 600000
smallfile = None
f = 'bulk2dets4negs_1bulk.txt'
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