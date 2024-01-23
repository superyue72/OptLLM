from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size, num_layers, num_class, hidden_size, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.mlp = [
            nn.ModuleList(
                
            )
        ]
        self.linear = nn.Linear(hidden_size, num_class)
        self.criterion = criterion
    
    def forward(self, X, Y=None, device='cpu'):
        X = X.to(device)
        X = self.mlp1(X)
        X = self.dropout(X)
        X = self.mlp2(X)
        if Y is not None:


