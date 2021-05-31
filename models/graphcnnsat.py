import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from mlp import MLP

class GraphCNNSAT(nn.Module):

    def __init__(self, num_layers=5, num_mlp_layers=2, input_dim, hidden_dim=64, output_dim, final_dropout=0.5, random = True, maxclause = 10000, maxvar = 10000, half_compute = True, var_classification = True, clause_classification = False, device):

        super(GraphCNNSAT, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.random = random

        self.maxclause = maxclause
        self.maxvar = maxvar
        self.half_compute = half_compute

        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))


        self.var_classification = var_classification
        self.clause_classification = clause_classification

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def next_layer_eps(self, h_clause, h_var, layer, biggraph):
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)

        # graph should be (maxvar+maxclause)*batch_size x(maxvar+maxclause)*batch_size
        # h should be (maxvar+maxclause)*batch_size

        if self.half_compute:
            clause_pooled = torch.hspmm(biggraph, h_var)
            var_pooled = torch.hspmm(torch.transpose(biggraph), h_clause)

            if self.neighbor_pooling_type == "average":
                degree_clauses = torch.hspmm(biggraph, torch.ones((biggraph.shape[0], 1)).to(self.device))
                clause_pooled = clause_pooled/degree_clauses
                degree_vars = torch.hspmm(torch.transpose(biggraph), torch.ones((biggraph.shape[1], 1)).to(self.device))
                var_pooled = var_pooled/degree_vars

                pooled = torch.cat([clause_pooled, var_pooled])
        else:
            h = torch.cat([h_clause, h_var])
            pooled = torch.hspmm(biggraph, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(biggraph, torch.ones((biggraph.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        pooled = pooled + (1+self.eps[layer])*torch.cat([h_clause, h_var])
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return torch.split(h,self.maxclause)

    def forward(self, batch_graph, half = True):

        nclause = batch_graph[0].shape[0]
        nvar = batch_graph[0].shape[1]

        clause_arities = [torch.cat([torch.tensor(np.asarray(ssm.sum(axis=1).flatten())[0]),torch.zeros(self.maxclause-nclause,dtype=torch.int32)]) for ssm in batch_graph]
        var_arities = [torch.cat([torch.tensor(np.asarray(ssm.sum(axis=0).flatten())[0]),torch.zeros(self.maxvar-nvar,dtype=torch.int32)]) for ssm in batch_graph]

        clause_feat = torch.cat(clause_arities, 0)
        var_feat = torch.car(var_arities,0)

        if self.random:
            r1 = torch.randint(self.random, size=(len(clause_feat), 1)).float() / self.random
            clause_feat = torch.cat([clause_feat, r1],1).to(self.device)
            r2 = torch.randint(self.random, size=(len(var_feat), 1)).float() / self.random
            var_feat = torch.cat([var_feat, r],1).to(self.device)

        if self.graph_embedding:
            clause_hidden_rep = [clause_feat]
            var_hidden_rep = [var_feat]

        h_clause = clause_feat
        h_var = var_feat

        biggraph = big_tensor_from_batch_graph(batch_graph)
        # batch_graph is a list of batch_size * (nclause x nvar) matrices
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)

        for layer in range(self.num_layers-1):
            h_clause, h_var  = self.next_layer_eps(h_clause,h_var, layer, biggraph)
            if graph_embedding:
                clause_hidden_rep.append(h_clause)
                var_hidden_rep.append(h_var)

        if self.var_classification:
            if self.clause_classification:
                return torch.softmax(self.fc1(torch.cat([h_clause,h_var],axis=0), 1))
            return torch.softmax(self.fc1(h_var, 1))
        if self.clause_classification:
            return torch.softmax(self.fc1(h_clause), 1)

        #below graph_embedding only

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

    def big_tensor_from_batch_graph(graphs):
        #batch graph is maxclause*batch_size x maxvar*batch_size
        for g in graphs:
            g.resize(self.maxclause,self.maxvar)
        big_mat = scipy.sparse.block_diag(graphs,format="coo", dtype= np.bool)
        if not self.half_compute:
            big_mat = scipy.sparse.block_diag([big_mat, big_mat.transpose])
        big_tensor = torch.sparse_coo_tensor([big_mat.row, big_mat.col], big_mat.data, big_mat.shape, dtype=torch.int32))
