import torch
import json


def load_data(file_path):
    result = list()
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    for i in data:
        temp_result = list()
        tokens = i['tokens']
        temp_result.append(tokens)
        temp_label = ['O'] * len(tokens)
        entities = i['entities']
        for e in entities:
            e_type = e['type']
            e_start = e['start']
            e_end = e['end']
            temp_label[e_start] = 'B-' + e_type
            temp_label[e_start + 1: e_end] = ['I-' + e_type] * (e_end - e_start - 1)
        temp_result.append(temp_label)
        result.append(temp_result)
    return result


def preprocess(text, to_ix):
    indexes = [to_ix[i] for i in text]
    return torch.tensor([indexes], dtype=torch.long)


class Evaluator:

    def __init__(self, model, testing_data, word_to_ix, tag_to_ix):
        self.model = model
        self.testing_data = testing_data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def evaluate(self):
        with torch.no_grad():
            label_score = [
                {
                    'tp': 0,
                    'fp': 0,
                    'tn': 0,
                    'fn': 0,
                }
                for _ in range(len(self.tag_to_ix))
            ]

            for sentence, labels in self.testing_data:
                inputs = preprocess(sentence, self.word_to_ix)
                targets = torch.squeeze(preprocess(labels, self.tag_to_ix), 0)
                prediction = torch.squeeze(self.model(inputs), 0)
                # calculate tp, fp, tn and fn in each kind of label
                for pred, tag in zip(prediction, targets):
                    if pred == tag:
                        for i, _ in enumerate(label_score):
                            if i == tag:
                                label_score[i]['tp'] += 1
                            else:
                                label_score[i]['tn'] += 1
                    else:
                        for i, _ in enumerate(label_score):
                            if i == pred:
                                label_score[i]['fp'] += 1
                            elif i == tag:
                                label_score[i]['fn'] += 1
                            else:
                                label_score[i]['tn'] += 1
            # sum all kinds of labels' accuracy, precision, recall, f1 as the final score
            total_a = 0
            total_p = 0
            total_r = 0
            total_f1 = 0
            cnt = 0
            for i, _ in enumerate(label_score):
                tp = label_score[i]['tp']
                fp = label_score[i]['fp']
                tn = label_score[i]['tn']
                fn = label_score[i]['fn']
                tmp_a = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0
                tmp_p = tp / (tp + fp) if tp + fp else 0
                tmp_r = tp / (tp + fn) if tp + fn else 0
                tmp_f1 = 2 * tmp_p * tmp_r / (tmp_p + tmp_r) if tmp_p + tmp_r else 0
                total_a += tmp_a
                total_p += tmp_p
                total_r += tmp_r
                total_f1 += tmp_f1
                cnt += 1
            self.accuracy = total_a / cnt
            self.precision = total_p / cnt
            self.recall = total_r / cnt
            self.f1 = total_f1 / cnt
