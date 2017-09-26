
#from debugHelper import DebugHelpers as d
import re
import pickle
import nltk
import textwrap

class Data():
    def __init__(self, tree_data=None, word_dict=None, relation_list=None):
        self.tree_data = [] if (tree_data is None) else tree_data
        self.word_dict = dict() if (word_dict is None) else word_dict
        self.relation_list = [] if (relation_list is None) else relation_list
        self.word_list = []

    # Reads train/test data from file, parses as nltk, adds labels for inner nodes ('cps')
    # and leaf nodes ('.')
    def loadData(self, data_file, data_type_str, constr_dict,
                 print_level_data=0, separator_char='\t', cpr_layer_bool=1):

        tl = "Loading data for: "+data_type_str
        #d.pl(60, tl, pad_before=1)

        if constr_dict:
            print("Dict. scope: Local and global")
        else:
            print("Dict. scope: Local only")

        print("Data file:  ", data_file)
        with open(data_file, 'r') as f:
            trees = []
            relset = set()
            wordset = set()
            for line in f:

                if cpr_layer_bool:
                    relation, s1, s2  = line.split(separator_char)
                    relation = relation.strip() # remove initial/ending whitespace
                    relset = relset.union({relation})

                    # Step 1: '.' before words i.e. leaf nodes
                    s1 = re.sub(r"([^()\s]+)", r"(. \1)", s1)
                    s2 = re.sub(r"([^()\s]+)", r"(. \1)", s2)
                    # Step 2: Label 'cps' after brackets not followed by '.', then: nltk tree
                    t1 = nltk.tree.Tree.fromstring(re.sub(r"\( ", r"(cps ", s1))
                    t2 = nltk.tree.Tree.fromstring(re.sub(r"\( ", r"(cps ", s2))

                    trees += [t1, t2]
                    self.tree_data += [(relation, t1, t2)]

                    wordset = wordset.union(set(t1.leaves()))
                    wordset = wordset.union(set(t2.leaves()))

                else:
                    relation, s1 = line.split(separator_char)
                    relation = relation.strip()
                    relset = relset.union({relation})

                    s1 = re.sub(r"([^()\s]+)", r"(. \1)", s1)
                    t1 = nltk.tree.Tree.fromstring(re.sub(r"\( ", r"(cps ", s1))

                    trees += [t1]
                    self.tree_data += [(relation, t1)]

                    wordset = wordset.union(set(t1.leaves()))

            relation_list = sorted(relset)

            wordlist = sorted(wordset)
            self.word_list = wordlist
            word_dict = {i:j for j,i in enumerate(wordlist)}

            print("Total pairs:", len(self.tree_data))

            if constr_dict:
                print("Dictionary: ", len(word_dict))
                print("Relations:  ", len(relation_list))
                #param.total_word_num = len(wordset)
                #param.num_rels = len(relation_list)
            else:
                print("Dictionary: ", len(word_dict))
                print("Relations:  ", len(relation_list))

            if print_level_data >= 2:
                print("\n> word_dict")
                print(word_dict, "\n")
                print("> word_dict (sorted)")
                print(sorted([(x, y) for x, y in zip(word_dict.values(), word_dict.keys())]), "\n")
            if print_level_data >= 1:
                print("\n> Vocabulary (sorted)")
                vocab_sorted = str(sorted(list(wordset)))
                str_out = textwrap.wrap(vocab_sorted, width=80)
                for sub_str in str_out:
                    print(sub_str)
                print("\n> Relations")
                str_out = textwrap.wrap(str(relation_list), width=80)
                for sub_str in str_out:
                    print(sub_str)
                #input("Done?")
            if print_level_data >= 3:
                print("\n> Full dataset")
                for i in self.tree_data:
                    print(i[0])
                    print(i[1])
                    if cpr_layer_bool: print(i[2])
                    print("")
                #input("Done?")

        if constr_dict:
            self.word_dict = word_dict
            self.relation_list = relation_list
            return (self.word_dict, self.relation_list)