import torch
import torch.nn as nn
import torch.nn.functional as func


class Model(nn.Module):

    def __init__(self, vocab_size, label_num, embedding_size, hidden_size=100, dropout=0.2):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, 64, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(1, 10, 3)
        self.hidden2label = nn.Linear(hidden_size, label_num)

    def forward(self, x):
        emb = self.embedding(x)  # (B, L) -> (B, L, emb_size)
        emb = self.dropout(emb)
        lstm_out, _ = self.lstm(emb)  # (B, L, emb_size) -> (B, L, 128)
        conv_out = self.conv(lstm_out)  # (B, L, 128) -> (B, L, hidden_size)
        out = self.hidden2label(conv_out)  # (B, L, H) -> (B, L, out_size)
        out = func.log_softmax(out, dim=-1)
        return out


if __name__ == '__main__':
    model = Model(200, 2, 100)
    model.forward(torch.tensor([
        [128, 192, 666]
    ]))

