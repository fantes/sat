import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP

class GraphCNNSAT(nn.Module):

    def __init__(self, num_layers=5, num_mlp_layers=2, input_dim, hidden_dim=64, output_dim, final_dropout=0.5, random = True, device):

        super(GraphCNNSAT, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.random = random


        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.fc1 = nn.Linear(hidden_dim, output_dim)

        #TODO/GUILLO define self.graph_pool for whole graph embedding
        # will be block "diagonal" matrix of either ones (sum) or ones/numel (average)
        # depending on how h is flattened

    def forward(self, (batch_graph,var_arities, clauses_arities)):
        feat = var_arities.extend(clauses_arities)

        if self.random:
            r = torch.randint(self.random, size=(len(feat), 1)).float() / self.random
            feat = torch.cat([feat, r],1).to(self.device)

        hidden_rep = [feat]

        h = feat

        for layer in range(self.num_layers-1):
            h = self.next_layer_eps(h, layer, batch_graph)
            hidden_rep.append(h)

        if self.node_classification:
            return torch.softmax(self.fc1(h), 1)

        score_over_layer = 0

        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            #TODO/GUILLO : SPMM takes a matrix, h should also be converted to a matrix
            # ie flattened / filled with zeros
            # graphpool is filled with 1 on corresponding coordinates for sum pooling
            # or 1/numel for average pooling
            pooled_h = torch.spmm(self.graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer
