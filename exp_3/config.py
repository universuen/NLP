import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 300  # 由于标签一共有B\I\O\START\STOP 5个，所以embedding_dim为5
HIDDEN_DIM = 256  # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2
