import torch
import torch.nn as nn
import torch.nn.functional as func


class Model(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_size, hidden_size, dropout=0.2):
        super(Model, self).__init__()
        self.name = "BiLSTM"
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(embedding_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.hidden2label = nn.Linear(hidden_size, len(tag_to_ix))
        # self.tag_to_ix = tag_to_ix

    def criterion(self, x, targets):
        criterion = nn.NLLLoss()
        predictions = self._forward(x)
        loss = 0
        for pred, tag in zip(predictions, targets):
            loss += criterion(pred, tag)
        return loss

    def _forward(self, x):
        emb = self.embedding(x)  # (B, L) -> (B, L, emb_size)
        emb = self.dropout(emb)
        lstm_out, _ = self.lstm(emb)  # (B, L, emb_size) -> (B, L, H)
        out = self.hidden2label(lstm_out)  # (B, L, H) -> (B, L, out_size)
        out = func.log_softmax(out, dim=-1)
        return out

    def forward(self, x):
        return torch.max(self._forward(x), -1)[1]

