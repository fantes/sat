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

    def next_layer_eps(self, h, layer, graph):
        # graph should be (maxvar+maxclause)*batch_size x(maxvar+maxclause)*batch_size
        # h should be (maxvar+maxclause)*batch_size
        pooled = torch.spmm(graph,h)
        if self.neighbor_pooling_type == "average":
            degree = torch.spmm(graph, torch.ones((graph.shape[0], 1)).to(self.device))
            pooled = pooled/degree
        pooled = pooled + (1+self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self, batch_graph, half = True):

        nclause = batch_graph[0].shape[0]
        nvar = batch_graph[0].shape[1]

        clause_arities = [np.asarray(ssm.sum(axis=1)).flatten().resize(maxclause) for ssm in batch_graph]
        var_arities = [np.asarray(ssm.sum(axis=0)).flatten().resize(maxvar) for ssm in batch_graph]

        clause_feat = torch.cat(clause_arities, 0)
        var_arities = torch.car(var_arities,0)

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

def batch_graph_from_graphlist(graphs,maxclause, maxvar, half = False):
    for g in graphs:
        g.resize(maxclause,maxvar)
    if half:
        mats = graphs
    else
        mats = []
        for g in graphs:
            mats.extend([g,g.transpose()])
    big_mat = scipy.sparse.block_diag(mats,format="coo", dtype= np.bool)
    big_tensor = torch.sparse_coo_tensor([big_mat.row, big_mat.col], big_mat.data, big_mat.shape, dtype=torch.bool))
