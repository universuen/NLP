from exp_3.models import BiLSTM, BiLSTM_CRF

"""Data Path"""

TRAINING_DATA = "./conll04/conll04_train.json"
TESTING_DATA = "./conll04/conll04_test.json"

"""Model Parameters"""

MODEL = BiLSTM.Model
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 256
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4
EPOCHS = 1
