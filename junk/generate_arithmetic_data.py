import random
from itertools import product

class ArithmeticDataGenerator():
    def __init__(self, max_int, train_nr, test_nr, expression_length):
        self.max_int = max_int
        self.train_nr = train_nr
        self.test_nr = test_nr
        self.total_nr = self.train_nr + self.test_nr
        self.exp_length = expression_length
        self.ints = [i for i in range(max_int)]
        self.ratio = (self.total_nr * 1.0) / ( ( ( self.max_int ** self.exp_length ) ** 2) )


    def all_pairs_shuffled(self):
        all_expressions = product(self.ints, repeat=self.exp_length)
        all_pairs = product(all_expressions, all_expressions)
        random.shuffle(all_pairs)
        return(all_pairs)

    # def all_expressions(self):
    #     all_expressions = product(self.ints, repeat=self.exp_length)
    #     #print(all_expressions)
    #     return(all_expressions)

    def possible_bracket_locations(self):
        nr_bracket_pairs = self.exp_length - 1
        locations = []

        for i in range(nr_bracket_pairs):
            if i == 0:
                # spanning brackets obligatory
                    locations += [0, self.exp_length]
            else:
                pass



    def generate_all(self):

        for idx in range(self.total_nr):
            left = random.sample(self.ints, self.exp_length)
            right = random.sample(self.ints, self.exp_length)

            print(left)
            print(right)

            # random brackets

            # check if not generated already



# a = ArithmeticDataGenerator(4, 10, 2, 3)
# a.generate_all()

def allbinarytrees(s):
    if len(s) == 1:
        yield s
    else:
        for i in range(1, len(s), 2):
            for l in allbinarytrees(s[:i]):
                for r in allbinarytrees(s[i+1:]):
                    yield '({}{}{})'.format(l, s[i], r)

for i in allbinarytrees('1+2+3'):
    print(i)