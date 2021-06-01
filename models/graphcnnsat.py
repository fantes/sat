import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
sys.path.append("../data/")
from graphDataset import GraphDataset
from mlp import MLP

import numpy as np
import scipy.sparse

class GraphCNNSAT(nn.Module):

    def __init__(self, num_layers=10, num_mlp_layers=2, input_dim=1, hidden_dim=64, output_dim=2, final_dropout=0.5, random = 1, maxclause = 10, maxvar = 20, half_compute = True, var_classification = True, clause_classification = False, graph_embedding = False, neighbor_pooling_type = "average", graph_pooling_type = "average", device=torch.device("cuda:0")):

        super(GraphCNNSAT, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.random = random
        self.input_dim = input_dim

        self.graph_embedding = graph_embedding

        if random:
            self.input_dim = self.input_dim + random
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.maxclause = maxclause
        self.maxvar = maxvar
        self.half_compute = half_compute

        self.neighbor_pooling_type = neighbor_pooling_type

        self.graph_pooling_type = graph_pooling_type

        self.eps = nn.Parameter(torch.ones(self.num_layers-1))

        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.num_mlp_layers = num_mlp_layers

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(self.num_mlp_layers, self.input_dim, self.hidden_dim, self.hidden_dim))
            else:
                self.mlps.append(MLP(self.num_mlp_layers, self.hidden_dim, self.hidden_dim, self.hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.mlps.to(self.device)
        self.batch_norms.to(self.device)

        self.var_classification = var_classification
        self.clause_classification = clause_classification

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(self.input_dim, self.output_dim))
            else:
                self.linears_prediction.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim).to(device)
        self.final_dropout = final_dropout


    def big_tensor_from_batch_graph(self, graphs):
        #batch graph is maxclause*batch_size x maxvar*batch_size
        for g in graphs:
            g.resize(self.maxclause,self.maxvar)
        big_mat = scipy.sparse.block_diag(graphs,format="coo", dtype= np.bool)
        if not self.half_compute:
            big_mat = scipy.sparse.block_diag([big_mat, big_mat.transpose])
        big_tensor = torch.sparse_coo_tensor([big_mat.row, big_mat.col], big_mat.data, big_mat.shape, dtype=torch.int32)
        return big_tensor


    def build_graph_pooler(self,batch_size,nclause,nvar):
        blocks = []
        for i in range(batch_size):
            blocks.append(np.mat(np.ones(self.maxclause+self.maxvar)))
            if self.graph_pooling_type == "average":
                blocks[-1] = blocks[-1]/(nclause[i] + nvar[i])
        spgraphpooler = scipy.sparse.block_diag([blocks])
        self.graph_pooler = torch.sparse_coo_tensor([spgraphpooler.row,spgraphpooler.col],spgraphpooler.data, spgraphpooler.shape, dtype=torch.float32)


    def next_layer_eps(self, h_clause, h_var, layer, biggraph,batch_size):
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)

        # graph should be (maxvar+maxclause)*batch_size x(maxvar+maxclause)*batch_size
        # h should be (maxvar+maxclause)*batch_size


        if self.half_compute:
            clause_pooled = torch.hspmm(biggraph, h_var).to_dense()
            var_pooled = torch.hspmm(torch.transpose(biggraph,0,1), h_clause).to_dense()

            if self.neighbor_pooling_type == "average":
                degree_clauses = torch.hspmm(biggraph, torch.ones((biggraph.shape[1], 1)).to(self.device)).to_dense()
                clause_pooled = clause_pooled/degree_clauses
                degree_vars = torch.hspmm(torch.transpose(biggraph,0,1), torch.ones((biggraph.shape[0], 1)).to(self.device)).to_dense()
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
        return torch.split(h,[self.maxclause*batch_size,self.maxvar*batch_size])

    def forward(self, batch_graph, half = True):


        clause_arities = []
        var_arities = []
        nclause = []
        nvar = []
        batch_size = len(batch_graph)

        for ssm in batch_graph:
            nclause.append(ssm.shape[0])
            nvar.append(ssm.shape[1])
            clause_arities.append(torch.cat([torch.tensor(np.asarray(ssm.sum(axis=1).flatten())[0]),torch.zeros(self.maxclause-nclause[-1],dtype=torch.int32)]))
            var_arities.append(torch.cat([torch.tensor(np.asarray(ssm.sum(axis=0).flatten())[0]),torch.zeros(self.maxvar-nvar[-1],dtype=torch.int32)]))

        clause_feat = torch.cat(clause_arities, 0)
        clause_feat.unsqueeze_(1)
        var_feat = torch.cat(var_arities,0)
        var_feat.unsqueeze_(1)

        if self.random > 0:
            r1 = torch.randint(self.random, size=(len(clause_feat),1)).float() / self.random
            clause_feat = torch.cat([clause_feat, r1],1).to(self.device)
            r2 = torch.randint(self.random, size=(len(var_feat), 1)).float() / self.random
            var_feat = torch.cat([var_feat, r2],1).to(self.device)

        if self.graph_embedding:
            clause_hidden_rep = [clause_feat]
            var_hidden_rep = [var_feat]

        h_clause = clause_feat.to(self.device)
        h_var = var_feat.to(self.device)

        biggraph = self.big_tensor_from_batch_graph(batch_graph).to(self.device).to(torch.float32)
        # batch_graph is a list of batch_size * (nclause x nvar) matrices
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)

        for layer in range(self.num_layers-1):
            h_clause, h_var  = self.next_layer_eps(h_clause,h_var, layer, biggraph,batch_size)
            if self.graph_embedding:
                clause_hidden_rep.append(h_clause)
                var_hidden_rep.append(h_var)



        if self.var_classification:
            if self.clause_classification:
                return torch.softmax(self.fc1(torch.cat([h_clause,h_var],axis=0)), 1)
            return torch.softmax(self.fc1(h_var), 1)
        if self.clause_classification:
            return torch.softmax(self.fc1(h_clause), 1)




        #below graph_embedding only

        score_over_layer = 0

        #could be moved in constructor for a  fixed batch_size
        graph_pooler = self.build_graph_pooler(batch_size, nclause,nvar)


        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(zip(h_clause,h_var)):
            h = torch.cat([h_clause, h_var])
            pooled_h = torch.spmm(self.graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer


def main():
    model = GraphCNNSAT()
    tds = GraphDataset('../data/test/ex.graph', cachesize=0, path_prefix="/home/infantes/code/sat/data/")
    ssm,labels = tds.getitem(0)
    batch_graph=[ssm,ssm]
    model.forward(batch_graph)

if __name__ == '__main__':
    main()
