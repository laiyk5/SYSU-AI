def load_attributes(path):
    attributes = list()
    continuous_indexes = list()
    with open(path) as f:
        for i in range(0, 96):
            f.readline()
        for i in range(96, 110):
            l = re.findall(r'[^:,\.\s]+', f.readline())
            if l[1:] == ['continuous']:
                continuous_indexes.append(len(attributes))
                attributes.append(Attribute(l[0], list()))
            else:
                attributes.append(Attribute(l[0], l[1:]))
    return attributes, continuous_indexes


def load_traning_examples(path, weighting):
    training_examples = list()
    with open(path) as f:
        line = f.readline()
        while line != '\n':
            l = re.findall(r'[^,\s]+', line)
            if weighting or '?' not in l:
                example = Example({attributes[i].name: l[i] for i in range(len(attributes))}, l[-1])
                training_examples.append(example)
            line = f.readline()
    return training_examples


# decision tree需要你们自己用训练集先训练好，然后作为参数传入, 
# decision_tree_predicting需要自己实现，根据带预测样本的属性和训练好的决策树预测该样本工资属性是否 >50K
def testing(path, decision_tree, continuous_indexes, continuous_mid, attributes):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    positive = None
    with open(path) as f:
        f.readline()
        line = f.readline()
        while line != '\n':
            l = re.findall(r'[^,.\s]+', line)
            example_attributes = {attributes[i].name: l[:-1][i] for i in range(len(attributes))}
            for index in continuous_indexes:
                i = 0
                while i < len(continuous_mid[index]) and float(l[index]) > continuous_mid[index][i]:
                    i += 1
                example_attributes[attributes[index].name] = str(i)
            if positive is None:
                positive = l[-1]
            for classification, weight in decision_tree_predicting(example_attributes, decision_tree):
                if l[-1] == positive:
                    if classification == positive:
                        TP += weight
                    else:
                        FP += weight
                else:
                    if classification != positive:
                        TN += weight
                    else:
                        FN += weight
            line = f.readline()
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)