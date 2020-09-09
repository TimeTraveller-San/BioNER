import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,
                vocab_size,
                embedding_dim,
                lstm_hidden_dim,
                number_of_tags,
                n_stack=2,
                dropout=0.2,
                bidirectional=True):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            lstm_hidden_dim, 
                            batch_first=True,
                            num_layers=n_stack,
                            dropout=dropout,
                            bidirectional=bidirectional)
        if bidirectional: lstm_hidden_dim *= 2
        self.fc = nn.Linear(lstm_hidden_dim, number_of_tags)

    def forward(self, s):
        s = self.embedding(s)   
        s, _ = self.lstm(s)           
        s = s.reshape(-1, s.shape[2])
        s = self.fc(s)
        return F.log_softmax(s, dim=1)   