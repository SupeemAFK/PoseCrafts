import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        self.rnn1 = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn2 = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn3 = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn4 = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn5 = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.o = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding):
        o_n1, h_n1 = self.rnn1(embedding)
        o_n2, h_n2 = self.rnn2(o_n1, h_n1)
        o_n3, h_n3 = self.rnn3(o_n2, h_n2)
        o_n4, h_n4 = self.rnn4(o_n3, h_n3)
        o_n5, h_n5 = self.rnn5(o_n4, h_n4)
        output = self.o(o_n5)
        return output