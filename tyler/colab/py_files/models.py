import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)


class lstm_clf(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim, hidden, weight, max_len, device, p_dropout):
        super(lstm_clf, self).__init__()
        self.device = device
        self.hidden_dim = hidden
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim * max_len, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, num_labels)
        self.dropout = nn.Dropout(p=p_dropout)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight))
        self.embedding.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden = self.init_hidden()

    def forward(self, inputs, h):
        x = inputs.view(1, -1)
        x = x.view(1, -1, self.embedding_dim)
        lstm_out, h = self.lstm(x, h)

        x = self.linear_2(lstm_out[:, -1])
        x = self.dropout(x)
        probs = F.softmax((x), dim=1)
        return probs[0], h

    def init_hidden(self):
        batch_size = 1
        h0 = torch.zeros((1, batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((1, batch_size, self.hidden_dim)).to(self.device)
        hidden = (h0, c0)
        return hidden