
def diagnostic_labels(input, hypothesis):
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

    elif hypothesis == 'length':
        for sentence in input:
            sentence_labels = []
            length = 0
            for word in sentence:
                length += 1
                sentence_labels += [length]
            labels += [sentence_labels]

    elif hypothesis == 'pos':
        pos_tags = ['bracket', 'noun', 'verb', 'quant', 'neg']
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

    return (labels)