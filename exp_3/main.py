import json
from exp_3.models import BiLSTM
from exp_3.config import *


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
    return result[:50]


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


if __name__ == '__main__':
    training_data = load_data("./conll04/conll04_train.json")
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = {
        'B-Loc': 0,
        'I-Loc': 1,
        'B-Org': 2,
        'I-Org': 3,
        'B-Peop': 4,
        'I-Peop': 5,
        'B-Other': 6,
        'I-Other': 7,
        'O': 8,
        START_TAG: 9,
        STOP_TAG: 10
    }
    model = BiLSTM.Model(
        vocab_size=len(word_to_ix),
        tag_to_ix=tag_to_ix,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # if torch.cuda.is_available():
    #     model = model.cuda()

    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        print(epoch)
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

    # Check predictions after training
    with torch.no_grad():
        for i in range(10):
            precheck_sent = prepare_sequence(training_data[i][0], word_to_ix)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[i][1]], dtype=torch.long)
            print(model(precheck_sent))
            print(precheck_tags)