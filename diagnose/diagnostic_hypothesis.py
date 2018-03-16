
def diagnostic_labels(input, hypothesis):
    quants = ['some', 'all']
    nouns = ['Romans', 'Italians', 'Europeans', 'Germans', 'children']
    verbs = ['fear', 'hate', 'like', 'love']

    labels = []

    if hypothesis == 'brackets':
        for sentence in input:
            sentence_labels = []
            depth = 0
            for word in sentence:
                if word == '(':
                    depth += 1
                elif word == ')':
                    depth -= 1
                sentence_labels += [depth]
            labels += [sentence_labels]

    elif hypothesis == 'recursive_depth':
        # determine recursive depth when there are no brackets in data

        for sentence in input:
            sentence_labels = []
            idx = 0
            if sentence[idx] in quants:
                sentence_labels += [2]
                idx += 1
            elif sentence[idx] == 'not':
                sentence_labels += [3, 3]
                idx += 2

            if sentence[idx] in nouns:
                sentence_labels += [2]
                idx += 1
            elif sentence[idx] == 'not':
                sentence_labels += [3, 3]
                idx += 2

            if sentence[idx] in verbs:
                sentence_labels += [2]
                idx += 1
            elif sentence[idx] == 'not':
                sentence_labels += [3, 3]
                idx += 2

            if sentence[idx] in quants:
                sentence_labels += [3]
                idx += 1

            if sentence[idx] in nouns:
                sentence_labels += [3]
            elif sentence[idx] == 'not':
                sentence_labels += [4, 4]

            labels += [sentence_labels]

    elif hypothesis == 'length':
        for sentence in input:
            sentence_labels = []
            length = 0
            for word in sentence:
                length += 1
                sentence_labels += [length]
            labels += [sentence_labels]

    elif hypothesis == 'pos':
        #pos_tags = ['bracket', 'noun', 'verb', 'quant', 'neg']
        pos_tags = ['quant', 'noun', 'verb', 'neg']
        pos_mapping = {pos_tag: idx for idx, pos_tag in enumerate(pos_tags)}

        for sentence in input:
            sentence_labels = []
            for word in sentence:
                if word in ['(', ')']:
                    sentence_labels += [pos_mapping['bracket']]
                elif word in ['Europeans', 'Germans', 'Italians', 'Romans', 'children']:
                    sentence_labels += [pos_mapping['noun']]
                elif word in ['fear', 'hate', 'like', 'love']:
                    sentence_labels += [pos_mapping['verb']]
                elif word in ['all', 'some']:
                    sentence_labels += [pos_mapping['quant']]
                elif word in ['not']:
                    sentence_labels += [pos_mapping['neg']]
            labels += [sentence_labels]

    elif hypothesis == 'monotonicity_direction':
        upward = 2
        downward = 0
        neutral = 1
        for sentence in input:
            sentence_labels = []
            idx = 0

            # quant1 = some
            if sentence[idx] == 'some':
                sentence_labels += [neutral]
                idx += 1
                if sentence[idx] == 'not':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] in nouns:
                        sentence_labels += [downward]
                        idx += 1
                elif sentence[idx] in nouns:
                    sentence_labels += [upward]
                    idx += 1

                # VP
                if sentence[idx] == 'not':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] in verbs:
                        sentence_labels += [downward]
                        idx += 1
                elif sentence[idx] in verbs:
                    sentence_labels += [upward]
                    idx += 1

                if sentence[idx] == 'some':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in nouns:
                            sentence_labels += [downward]
                    elif sentence[idx] in nouns:
                        sentence_labels += [upward]
                elif sentence[idx] == 'all':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in nouns:
                            sentence_labels += [upward]
                    elif sentence[idx] in nouns:
                        sentence_labels += [downward]

            # quant1 = all
            elif sentence[idx] == 'all':
                sentence_labels += [neutral]
                idx += 1
                # negated noun1
                if sentence[idx] == 'not':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] in nouns:
                        sentence_labels += [upward]
                        idx += 1
                # unnegated noun1
                elif sentence[idx] in nouns:
                    sentence_labels += [downward]
                    idx += 1

                # negated verb
                if sentence[idx] == 'not':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] in verbs:
                        sentence_labels += [downward]
                        idx += 1
                # unnegated verb
                elif sentence[idx] in verbs:
                    sentence_labels += [upward]
                    idx += 1

                # quant2 = some
                if sentence[idx] == 'some':
                    sentence_labels += [neutral]
                    idx += 1
                    # negated noun2
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in nouns:
                            sentence_labels += [downward]
                    # unnegated noun2
                    elif sentence[idx] in nouns:
                        sentence_labels += [upward]

                # quant2 = all
                elif sentence[idx] == 'all':
                    sentence_labels += [neutral]
                    idx += 1
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in nouns:
                            sentence_labels += [upward]
                    elif sentence[idx] in nouns:
                        sentence_labels += [downward]


            # quant1
            elif sentence[idx] == 'not':
                sentence_labels += [neutral]
                idx += 1
                if sentence[idx] == 'some':
                    sentence_labels += [neutral]
                    idx += 1

                    # negated noun1
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in nouns:
                            sentence_labels += [upward]
                            idx += 1
                    # unnegated noun1
                    elif sentence[idx] in nouns:
                        sentence_labels += [downward]
                        idx += 1

                    # negated verb
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in verbs:
                            sentence_labels += [upward]
                            idx += 1
                    # unnegated verb
                    elif sentence[idx] in verbs:
                        sentence_labels += [downward]
                        idx += 1

                    # quant2 = some
                    if sentence[idx] == 'some':
                        sentence_labels += [neutral]
                        idx += 1
                        # negated noun2
                        if sentence[idx] == 'not':
                            sentence_labels += [neutral]
                            idx += 1
                            if sentence[idx] in nouns:
                                sentence_labels += [upward]
                        # unnegated noun2
                        elif sentence[idx] in nouns:
                            sentence_labels += [downward]

                    # quant2 = all
                    elif sentence[idx] == 'all':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] == 'not':
                            sentence_labels += [neutral]
                            idx += 1
                            if sentence[idx] in nouns:
                                sentence_labels += [downward]
                        elif sentence[idx] in nouns:
                            sentence_labels += [upward]

                elif sentence[idx] == 'all':
                    sentence_labels += [neutral]
                    idx += 1

                    # negated noun1
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in nouns:
                            sentence_labels += [downward]
                            idx += 1
                    # unnegated noun1
                    elif sentence[idx] in nouns:
                        sentence_labels += [upward]
                        idx += 1

                    # negated verb
                    if sentence[idx] == 'not':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] in verbs:
                            sentence_labels += [upward]
                            idx += 1
                    # unnegated verb
                    elif sentence[idx] in verbs:
                        sentence_labels += [downward]
                        idx += 1

                    # quant2 = some
                    if sentence[idx] == 'some':
                        sentence_labels += [neutral]
                        idx += 1
                        # negated noun2
                        if sentence[idx] == 'not':
                            sentence_labels += [neutral]
                            idx += 1
                            if sentence[idx] in nouns:
                                sentence_labels += [upward]
                        # unnegated noun2
                        elif sentence[idx] in nouns:
                            sentence_labels += [downward]

                    # quant2 = all
                    elif sentence[idx] == 'all':
                        sentence_labels += [neutral]
                        idx += 1
                        if sentence[idx] == 'not':
                            sentence_labels += [neutral]
                            idx += 1
                            if sentence[idx] in nouns:
                                sentence_labels += [downward]
                        elif sentence[idx] in nouns:
                            sentence_labels += [upward]

            labels += [sentence_labels]

    elif hypothesis == 'negation_scope':
        for sentence in input:
            sentence_labels = []
            for idx, word in enumerate(sentence):
                if idx == 0:
                    sentence_labels += [0]
                else:
                    if sentence[idx - 1] == 'not':
                        sentence_labels += [1]
                    else:
                        sentence_labels += [0]
            labels += [sentence_labels]

    return (labels)