import torch.optim as optim
from exp_3.config import *
from exp_3.utilities import *


if __name__ == '__main__':
    """load the data"""

    training_data = load_data(TRAINING_DATA)
    testing_data = load_data(TESTING_DATA)

    """prepare necessary dictionaries"""

    # make word-to-index dictionary
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    for sentence, tags in testing_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # make tag-to-index dictionary
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
        # the below items are only used in BiLSTM_CRF
        '<START>': 9,
        '<STOP>': 10
    }

    """initialization"""

    # instantiate a model
    model = BiLSTM.Model(
        vocab_size=len(word_to_ix),
        tag_to_ix=tag_to_ix,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE
    )
    # instantiate a optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    # instantiate a evaluator
    evaluator = Evaluator(
        model=model,
        testing_data=testing_data,
        word_to_ix=word_to_ix,
        tag_to_ix=tag_to_ix
    )
    print(
        f"*****Parameters*****\n"
        f"> Model: {model.name}\n"
        f"> Embedding Size: {EMBEDDING_SIZE}\n"
        f"> Hidden Size: {HIDDEN_SIZE}\n"
        f"> Learning Rate: {LEARNING_RATE}\n"
        f"> Weight decay: {WEIGHT_DECAY}\n"
        f"> Epochs: {EPOCHS}\n"
        f"********************\n"
    )

    """train the model"""

    print('Started Training')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for sentence, tags in training_data:
            model.zero_grad()  # reset the gradient
            inputs = preprocess(sentence, word_to_ix)
            targets = preprocess(tags, tag_to_ix)
            loss = model.criterion(inputs, targets)  # criterion method includes the forward method
            loss.backward()  # calculate the gradient depending on the loss
            optimizer.step()  # apply the calculated gradient
            running_loss += loss.item()
        print(f"> epoch: {epoch + 1}, loss: {running_loss / len(training_data): .5f}")
    print('Finished Training\n')

    """evaluate the model"""

    print('Started Evaluating')
    evaluator.evaluate()  # calculate the evaluating scores
    print('Finished Evaluating\n')
    print(
        f"*****Evaluation*****\n"
        f"> Accuracy: {evaluator.accuracy:.5f}\n"
        f"> Precision: {evaluator.precision:.5f}\n"
        f"> Recall: {evaluator.recall:.5f}\n"
        f"> F1: {evaluator.f1:.5f}\n"
        f"********************\n"
    )
