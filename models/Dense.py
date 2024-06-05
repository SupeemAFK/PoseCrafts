import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, output_dim)

    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        output = self.o(x)
        return output