"""
scrape log files and average scores
"""

def average_scores(model):
    root='/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/' + model + '_binaryfol_train578_test69_' #1.txt'
    runs = ['1', '2', '3', '4', '5']
    training_sum = 0
    testing_sum = 0

    for run in runs:
        file_name = root + run + '.txt'
        with open(file_name, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 30:
                    testing_accuracy = line.split()[1]
                    testing_sum += float(testing_accuracy)

                if idx == 32:
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
