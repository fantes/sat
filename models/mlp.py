import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graphNorm import GraphNorm

###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, graph_norm, maxclause, maxvar):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        self.maxclause = maxclause
        self.maxvar = maxvar
        self.graph_norm = graph_norm
        self.hidden_dim = hidden_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim, dtype=torch.half))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            if self.graph_norm:
                for layer in range(num_layers - 1):
                    self.norms.append(GraphNorm(self.hidden_dim,self.maxclause,self.maxvar))
            else:
                for layer in range(num_layers - 1):
                    self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
