import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
sys.path.append("../data/")
sys.path.append("../utils/")
from utils import *
from graphDataset import GraphDataset
from mlp import MLP

import numpy as np
import scipy.sparse

class GraphCNNSAT(nn.Module):

    def __init__(self, num_layers=10, num_mlp_layers=2,  hidden_dim=8, output_dim=2, final_dropout=0.5, random = 1, maxclause = 10, maxvar = 20, half_compute = True, var_classification = True, clause_classification = False, graph_embedding = False, PGSO=False, mPGSO=False, neighbor_pooling_type = "average", graph_pooling_type = "average", lfa = True, device=torch.device("cuda:0")):

        super(GraphCNNSAT, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.random = random
        self.input_dim = 1 # arities

        self.PGSO = PGSO
        self.mPGSO = mPGSO

        self.graph_embedding = graph_embedding

        if random:
            self.input_dim = self.input_dim + random # could add multiple random values
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lfa = lfa #last fully adjacent layer
        self.maxclause = maxclause
        self.maxvar = maxvar
        self.half_compute = half_compute

        self.neighbor_pooling_type = neighbor_pooling_type

        self.graph_pooling_type = graph_pooling_type

        self.eps = nn.Parameter(torch.ones(self.num_layers-1))
        if self.mPGSO:
            self.a = nn.Parameter(torch.ones(self.num_layers-1))
            self.m1 = nn.Parameter(torch.ones(self.num_layers-1))
            self.m2 = nn.Parameter(torch.ones(self.num_layers-1))
            self.m3 = nn.Parameter(torch.ones(self.num_layers-1))
            self.e1 = nn.Parameter(torch.ones(self.num_layers-1))
            self.e2 = nn.Parameter(torch.ones(self.num_layers-1))
            self.e3 = nn.Parameter(torch.ones(self.num_layers-1))
            self.PGSO = True
        elif self.PGSO:
            self.a = nn.Parameter(torch.ones(1))
            self.m1 = nn.Parameter(torch.ones(1))
            self.m2 = nn.Parameter(torch.ones(1))
            self.m3 = nn.Parameter(torch.ones(1))
            self.e1 = nn.Parameter(torch.ones(1))
            self.e2 = nn.Parameter(torch.ones(1))
            self.e3 = nn.Parameter(torch.ones(1))


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
        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(self.input_dim, self.output_dim))
            else:
                self.linears_prediction.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.linears_prediction.to(self.device)

        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim).to(device)
        self.final_dropout = final_dropout

        if self.lfa:
            self.var_lfa = nn.Linear(self.maxvar, self.maxvar).to(device)
            self.clause_lfa = nn.Linear(self.maxclause, self.maxclause).to(device)


    def next_layer_eps(self, h_clause, h_var, layer, biggraph,batch_size):
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)
        # h should be (maxvar+maxclause)*batch_size

        # TODO see https://arxiv.org/abs/2101.10050 (page 5)
        # for an updated rule to learn in a larger adjacency operator space

        if self.neighbor_pooling_type == "average" or self.PGSO:
            if self.half_compute:
                degree_clauses = torch.hspmm(biggraph, torch.ones((biggraph.shape[1], 1)).to(self.device)).to_dense()
                degree_vars = torch.hspmm(torch.transpose(biggraph,0,1), torch.ones((biggraph.shape[0], 1)).to(self.device)).to_dense()
            else:
                degree = torch.hspmm(biggraph, torch.ones((biggraph.shape[0], 1)).to(self.device)).to_dense()
            degrees = torch.cat([degree_clauses, degree_vars])


        # how to weight efficiently every param in hspmm sum

        if self.PGSO:
            lindex = 0
        if self.mPGSO:
            lindex = layer
        if self.PGSO or self.mPGSO:
            la = self.a[lindex]
            le1 = self.e1[lindex]
            le2 = self.e2[lindex]
            le3 = self.e3[lindex]
            lm1 = self.m1[lindex]
            lm2 = self.m2[lindex]
            lm3 = self.m3[lindex]


        if self.half_compute:

            if self.PGSO:
                dje3 = torch.pow(degree_vars * la,le3)
                h_var_dje3 = torch.mul(h_var,dje3)
                clause_pooled = torch.hspmm(biggraph, h_var_dje3).to_dense()
                die2 = torch.pow(degree_clauses,le2)
                clause_pooled = torch.mul(clause_pooled, die2)

                dje3 = torch.pow(degree_clauses,le3)
                h_clause_dje3 = torch.mul(h_clause,dje3)
                var_pooled = torch.hspmm(torch.transpose(biggraph,0,1), h_clause_dje3).to_dense()
                die2 = torch.pow(degree_vars * la ,le2)
                var_pooled = torch.mul(var_pooled, die2)

            else:

                clause_pooled = torch.hspmm(biggraph, h_var).to_dense()
                var_pooled = torch.hspmm(torch.transpose(biggraph,0,1), h_clause).to_dense()


            if self.neighbor_pooling_type == "average": #should be subsumed by PGSO
                clause_pooled = clause_pooled/degree_clauses
                var_pooled = var_pooled/degree_vars

            pooled = torch.cat([clause_pooled, var_pooled])
        else:
            h = torch.cat([h_clause, h_var])

            if self.PGSO:
                dje3 = torch.pow(degrees * la, le3)
                h_dje3 = torch.mul(h,dje3)
                pooled = torch.hspmm(biggraph, h_dje3).to_dense()
                die2 = torch.pow(degrees, le2)
                pooled = torch.mul(pooled, die2)
            else:
                pooled = torch.hspmm(biggraph, h).to_dense()

            if self.neighbor_pooling_type == "average": #should be subsumed by graph_shift
                pooled = pooled/degree

        if self.PGSO:
            dai_e1 = torch.pow(degrees, le1)
            h = torch.cat([h_clause, h_var])
            pooled = (dai_e1 * lm1 * h) + lm3 * h + lm2 * pooled
        else:
            pooled = pooled + (1+self.eps[layer])*torch.cat([h_clause, h_var])
        pooled_rep = self.mlps[layer](pooled)
        #TODO add graphnorm, see https://arxiv.org/abs/2009.03294
        #https://github.com/lsj2408/GraphNorm
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)

        return torch.split(h,[self.maxclause*batch_size,self.maxvar*batch_size])


    def forward(self, batch_size, biggraph, clause_feat, var_feat, graph_pooler):

        if self.random > 0:
            for r in range(self.random):
                r1 = torch.randint(self.random, size=(len(clause_feat),1)).float() / self.random
                clause_feat = torch.cat([clause_feat, r1],1).to(self.device)
                r2 = torch.randint(self.random, size=(len(var_feat), 1)).float() / self.random
                var_feat = torch.cat([var_feat, r2],1).to(self.device)

        if self.graph_embedding:
            clause_hidden_rep = [clause_feat]
            var_hidden_rep = [var_feat]

        h_clause = clause_feat.to(self.device)
        h_var = var_feat.to(self.device)


        for layer in range(self.num_layers-1):
            h_clause, h_var  = self.next_layer_eps(h_clause,h_var, layer, biggraph,batch_size)
            if self.graph_embedding:
                clause_hidden_rep.append(h_clause)
                var_hidden_rep.append(h_var)

        # add last fully adjacent layer see https://arxiv.org/abs/2006.05205
        if self.lfa:
            batched_var = torch.reshape(h_var,(batch_size, self.maxvar, -1))
            lin_var = self.var_lfa(torch.transpose(batched_var,1,2))
            h_var = torch.reshape(torch.transpose(lin_var, 1,2),(batch_size*self.maxvar,-1))

            batched_clause = torch.reshape(h_var,(batch_size, self.maxclause, -1))
            lin_clause = self.clause_lfa(torch.transpose(batched_clause,1,2))
            h_clause = torch.reshape(torch.transpose(lin_clause, 1,2),(batch_size*self.maxclause,-1))

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

        #perform pooling over all nodes in each graph in every layer
        for layer, (h_clause,h_var) in enumerate(zip(clause_hidden_rep,var_hidden_rep)):
            h = torch.cat([h_clause, h_var])
            pooled_h = torch.spmm(graph_pooler, h)
            #below this can be considered as  page rank, see https://arxiv.org/abs/2006.07988
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer


def main():
    model = GraphCNNSAT(var_classification=True, clause_classification=False, graph_embedding = False, mPGSO=True)
    tds = GraphDataset('../data/test/ex.graph', cachesize=0, path_prefix="/home/infantes/code/sat/data/")
    m,labels = tds.getitem(0)['graph'], tds.getitem(0)['labels']
    batch_graph=[m,m,m]

    batch_size = len(batch_graph)

    clause_feat, var_feat, nclause, nvar = get_feat(batch_graph, model.maxclause, model.maxvar)

    #TODO : precompute degrees
    biggraph = big_tensor_from_batch_graph(batch_graph,model.maxclause,model.maxvar).to(model.device).to(torch.float32)

    model.forward(batch_size, biggraph, clause_feat, var_feat, None)

if __name__ == '__main__':
    main()
