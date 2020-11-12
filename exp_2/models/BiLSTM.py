import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.hidden2label = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L) -> (B, L, D)
        emb = self.dropout(emb)
        lstm_out, _ = self.lstm(emb)  # (B, L, D) -> (B, L, H)
        out = self.hidden2label(lstm_out[:, -1, :])  # (B, L, H) -> (B, H) -> (B, 2)
        out = F.log_softmax(out, dim=-1)
        return out


# if __name__ == '__main__':
#     a = [
#         [
#             [1, 1, 1],
#             [2, 2, 2]
#         ],
#         [
#             [3, 3, 3],
#             [4, 4, 4]
#         ]
#     ]
#     import torch
#     a = torch.tensor(a)
#     print(a[:, -1, :])
