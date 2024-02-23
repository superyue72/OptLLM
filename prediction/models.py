import torch
from torch import nn


class MLP(nn.Module):
    """ 
    A simple MLP with 1 hidden layer for multilabel (n_label) classification
    """
    def __init__(self, input_dim, hidden_dim, n_label, dropout=0.2, criterion=nn.BCELoss()):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_label)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.criterion = criterion

    def forward(self, x, y=None, device='cpu'):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        loss = None
        if y is not None:
            loss = self.criterion(x, y.to(device))
        return {
            "logits": x,
            "probabilities": x,
            "loss": loss,
        }
