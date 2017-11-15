
def combine(file_list, file_out):

    combined_data = set()

    for file in file_list:
        with open(file, 'r') as f:
            for idx, line in enumerate(f):
                #print(line.split())
                #if not 'two' in line.split():
                combined_data.add(line)

                # if idx == 10:
                #     break

    with open(file_out, 'w') as f:
        for line in combined_data:
            f.write(line)

    #return(combined_data)

#file_list = ['new2bulk.txt', 'new3bulk.txt', 'new4bulk.txt', 'new5bulk.txt', '/Users/Mathijs/Documents/School/MoL/thesis/big_files/bulk_binary_4negs.txt']

file_list = ['chunked_bulk2dets4negs_1bulk.txt_600000.txt',
             'chunked_bulk2dets4negs_2bulk.txt_600000.txt',
             'chunked_bulk2dets4negs_3bulk.txt_600000.txt',
             'chunked_bulk2dets4negs_4bulk.txt_600000.txt',
             'chunked_bulk2dets4negs_5bulk.txt_600000.txt',
             'chunked_bulk2dets4negs_6bulk.txt_600000.txt',
             'chunked_bulk2dets4negs_1bulk.txt_1200000.txt',
             'chunked_bulk2dets4negs_2bulk.txt_1200000.txt',
             'chunked_bulk2dets4negs_3bulk.txt_1200000.txt',
             'chunked_bulk2dets4negs_4bulk.txt_1200000.txt',
             'chunked_bulk2dets4negs_5bulk.txt_1200000.txt',
             'chunked_bulk2dets4negs_6bulk.txt_1200000.txt']
file_out = 'bulk_2dets_4negs_combined.txt'
combine(file_list, file_out)