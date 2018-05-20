# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class BiLSTM(nn.Module):
    """ PyTorch implementation of bi-directional lstm model.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.bi_lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.project = nn.Linear(self.hidden_size * self.num_layers, self.output_size)

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        return (h0, c0)

    def forward(self, X, hidden, seq_len=1):
        batch_size = X.size(0)
        embedding = self.embedding(X)
        lstm_output, hidden = self.bi_lstm(embedding.view(batch_size, seq_len, -1), hidden)
        output = self.project(lstm_output.view(batch_size, -1))
        return output, hidden




