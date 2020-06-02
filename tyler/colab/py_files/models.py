import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
from torch.nn.utils.rnn import pack_sequence,pad_sequence,pack_padded_sequence,pad_packed_sequence

class lstm_clf(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim, num_layers, hidden, batch_size, weight, max_len, device, p_dropout):
        super(lstm_clf, self).__init__()
        self.device = device
        self.hidden_dim = hidden
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim * max_len, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, num_labels)
        self.dropout = nn.Dropout(p=p_dropout)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight))
        self.embedding.requires_grad = False
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
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
        h0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)).to(self.device)
        hidden = (h0, c0)
        return hidden


class lstm_clf_batch(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim, num_layers, hidden, batch_size, weight, device, p_dropout):
        super(lstm_clf_batch, self).__init__()
        self.device = device
        self.hidden_dim = hidden
        self.embedding_dim = embedding_dim
        self.linear_2 = nn.Linear(self.hidden_dim, num_labels)
        self.dropout = nn.Dropout(p=p_dropout)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight),padding_idx=99999)
        self.embedding.requires_grad = False
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.hidden = self.init_hidden()

    def forward(self, padded, lengths, h):

        emb = self.embedding(padded)
        x = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        x, h = self.lstm(x, h)
        x, lengths_array = pad_packed_sequence(x, batch_first=True)

        last_seq_items = x[torch.arange(self.batch_size), lengths_array - 1]
        x = self.linear_2(last_seq_items)

        probs = F.softmax((x), dim=1)
        return probs, h

    def init_hidden(self):
        h0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.num_layers, self.batch_size, self.hidden_dim)).to(self.device)
        hidden = (h0, c0)
        return hidden