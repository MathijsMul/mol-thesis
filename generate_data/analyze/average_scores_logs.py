

def average_scores(model):

    #root = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/bowman_rep/logs/' #+ model + '/' #f1_1sumnn_bowman_rep.txt'
    root = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/' + model + '_binaryfol_0brackets_' #1.txt'
    #folds = ['f1', 'f2', 'f3', 'f4', 'f5']
    runs = ['1', '2', '3', '4', '5']

    training_sum = 0
    testing_sum = 0

    #for fold in folds:
    for run in runs:

        #'/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/bowman_rep/logs/sumnn/f1_1sumnn_bowman_rep.txt'

        #file_name = root + model + '/' + fold + '_' + run + model + '_bowman_rep.txt'
        file_name = root + run + '.txt'

        with open(file_name, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 79: #28:
                    testing_accuracy = line.split()[1]
                    testing_sum += float(testing_accuracy)

                if idx == 81: #30:
                    training_accuracy = line.split()[1]
                    training_sum += float(training_accuracy)

    avg_training_accuracy = float(training_sum) / 5
    avg_testing_accuracy = float(testing_sum) / 5

    print('Training: ', str(avg_training_accuracy))
    print('Testing: ', str(avg_testing_accuracy))

print('srn')
average_scores('srn')

print('gru')
average_scores('gru')

print('lstm')
average_scores('lstm')
