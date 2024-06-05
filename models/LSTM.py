import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.o = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding):
        o_n1, (h_n1, c_n1) = self.lstm1(embedding)
        o_n2, (h_n2, c_n2) = self.lstm2(o_n1, (h_n1, c_n1))
        o_n3, (h_n3, c_n3) = self.lstm3(o_n2, (h_n2, c_n2))
        o_n4, (h_n4, c_n4) = self.lstm4(o_n3, (h_n3, c_n3))
        o_n5, (h_n5, c_n5) = self.lstm5(o_n4, (h_n4, c_n4))
        output = self.o(o_n5)
        return output