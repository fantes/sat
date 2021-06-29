import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("../")
from data.graphDataset import *
from utils import *
import models.graphNorm
from models.mlp import MLP
from models.graphNorm import GraphNorm

import numpy as np
import scipy.sparse

class GraphCNNSAT(nn.Module):

    def __init__(self, num_layers=10, num_mlp_layers=2,  hidden_dim=2, output_dim=2, final_dropout=0.5, random = 1, maxclause = 10, maxvar = 20, graph_type = "var.var", var_classification = True, clause_classification = False, graph_embedding = False, PGSO=False, mPGSO=False, graph_norm = True, neighbor_pooling_type = "sum", graph_pooling_type = "average", lfa = True):


        super(GraphCNNSAT, self).__init__()

        self.final_dropout = final_dropout
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

        if graph_type == "var.var":
            self.varvar = True
        elif graph_type == "clause.var":
            self.varvar = False
        else:
            raise Exception("unkown graph type")

        self.lfa = lfa #last fully adjacent layer
        if self.varvar:
            self.maxclause = 0
            self.clause_classification = False
        else:
            self.maxclause = maxclause
            self.clause_classification = clause_classification
        self.maxvar = maxvar

        self.graph_norm = graph_norm

        self.neighbor_pooling_type = neighbor_pooling_type

        self.graph_pooling_type = graph_pooling_type

        self.eps = nn.Parameter(torch.ones(self.num_layers-1))
        if mPGSO:
            self.a = nn.Parameter(torch.ones(self.num_layers-1))
            self.m1 = nn.Parameter(torch.ones(self.num_layers-1))
            self.m2 = nn.Parameter(torch.ones(self.num_layers-1))
            self.m3 = nn.Parameter(torch.ones(self.num_layers-1))
            self.e1 = nn.Parameter(torch.ones(self.num_layers-1))
            self.e2 = nn.Parameter(torch.ones(self.num_layers-1))
            self.e3 = nn.Parameter(torch.ones(self.num_layers-1))
        elif PGSO:
            self.a = nn.Parameter(torch.ones(1))
            self.m1 = nn.Parameter(torch.ones(1))
            self.m2 = nn.Parameter(torch.ones(1))
            self.m3 = nn.Parameter(torch.ones(1))
            self.e1 = nn.Parameter(torch.ones(1))
            self.e2 = nn.Parameter(torch.ones(1))
            self.e3 = nn.Parameter(torch.ones(1))

        self.num_mlp_layers = num_mlp_layers

        self.embedder = MLP(self.num_mlp_layers, self.input_dim, self.hidden_dim, self.hidden_dim,self.graph_norm,self.maxclause,self.maxvar)


        self.mlps = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()


        for layer in range(self.num_layers-1):
            self.mlps.append(MLP(self.num_mlp_layers, self.hidden_dim, self.hidden_dim, self.hidden_dim,self.graph_norm,self.maxclause,self.maxvar))

            if self.graph_norm:
                self.norms.append(GraphNorm(self.hidden_dim,self.maxclause, self.maxvar))
            else:
                self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.var_classification = var_classification

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(self.input_dim, self.output_dim))
            else:
                self.linears_prediction.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.linears_prediction

        self.fc1 = nn.Linear(self.hidden_dim, self.output_dim, dtype=torch.float)
        self.final_dropout = final_dropout

        if self.lfa:
            self.var_lfa = nn.Linear(self.maxvar, self.maxvar)
            if not self.varvar:
                self.clause_lfa = nn.Linear(self.maxclause, self.maxclause)


    def print_mem(self, name, var):
        print(name+": " + str(var.nelement()*var.element_size()))
        print("shape: " + str(var.shape))

    def next_layer_eps(self, h_clause, h_var, layer, biggraph,batch_size):
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)
        # h should be (maxvar+maxclause)*batch_size

        # PGSO see https://arxiv.org/abs/2101.10050 (page 5)

        if self.neighbor_pooling_type == "average" or self.PGSO:
            if self.varvar:
                degrees = torch.sparse.mm(biggraph, torch.ones((biggraph.shape[0], 1)).to(biggraph.device)).to(torch.short).to(h_var.device)
            else:
                tempones = torch.ones((biggraph.shape[1], 1),device=biggraph.device, dtype=torch.float)
                degree_clauses = torch.sparse.mm(biggraph.to(torch.float), tempones).to(torch.short).to(h_var.device)
                tempones = torch.ones((biggraph.shape[0], 1),device=biggraph.device)
                degree_vars = torch.sparse.mm(torch.transpose(biggraph,0,1).to(torch.float), tempones).to(torch.short).to(h_var.device)
                degrees = torch.cat([degree_clauses, degree_vars])

        if self.mPGSO:
            lindex = layer
        elif self.PGSO:
            lindex = 0
        if self.PGSO or self.mPGSO:
            la = self.a[lindex]
            le1 = self.e1[lindex]
            le2 = self.e2[lindex]
            le3 = self.e3[lindex]
            lm1 = self.m1[lindex]
            lm2 = self.m2[lindex]
            lm3 = self.m3[lindex]


        if not self.varvar: #ie graph is clause x var

            if self.PGSO:
                dje3 = torch.pow(degree_vars * la,le3)
                h_var_dje3 = torch.mul(h_var,dje3)
                clause_pooled = torch.sparse.mm(biggraph.to(torch.float), h_var_dje3.to(torch.float).to(biggraph.device)).to(h_var.device)
                die2 = torch.pow(degree_clauses,le2.to(torch.float))
                clause_pooled = torch.mul(clause_pooled, die2)

                dje3 = torch.pow(degree_clauses,le3.to(torch.float))
                h_clause_dje3 = torch.mul(h_clause,dje3)
                var_pooled = torch.sparse.mm(torch.transpose(biggraph,0,1).to(torch.float), h_clause_dje3.to(torch.float).to(biggraph.device)).to(h_var.device)
                die2 = torch.pow(degree_vars * la ,le2.to(torch.float))
                var_pooled = torch.mul(var_pooled, die2)

            else:
                clause_pooled = torch.sparse.mm(biggraph.to(torch.float), h_var.to(torch.float).to(biggraph.device)).to(torch.float).to(h_var.device)
                var_pooled = torch.sparse.mm(torch.transpose(biggraph,0,1).to(torch.float), h_clause.to(torch.float).to(biggraph.device)).to(torch.float).to(h_var.device)

            if self.neighbor_pooling_type == "average": #should be subsumed by PGSO
                clause_pooled = clause_pooled/degree_clauses.expand_as(clause_pooled)
                var_pooled = var_pooled/degree_vars.expand_as(var_pooled)

            pooled = torch.cat([clause_pooled, var_pooled])
        else:
            h = h_var

            if self.PGSO:
                dje3 = torch.pow(degrees * la, le3)
                h_dje3 = torch.mul(h,dje3)
                pooled = torch.sparse.mm(biggraph, h_dje3.to(biggraph.device)).to(h_var.device)
                die2 = torch.pow(degrees, le2)
                pooled = torch.mul(pooled, die2)
            else:
                pooled = torch.sparse.mm(biggraph, h.to(biggraph.device)).to(h_var.device)

            if self.neighbor_pooling_type == "average": #should be subsumed by graph_shift
                pooled = pooled/degree.expand_as(pooled)

        if self.varvar:
            h = h_var
        else:
            h = torch.cat([h_clause, h_var])

        if self.PGSO:
            dai_e1 = torch.pow(degrees, le1.to(torch.float))
            pooled = (dai_e1 * lm1 * h) + lm3 * h + lm2 * pooled
        else:
            pooled +=  (1+self.eps[layer])*h

        pooled = F.relu(self.norms[layer](self.mlps[layer](pooled.to(torch.float))))

        if self.varvar:
            return None, pooled
        else:
            return torch.split(pooled,[self.maxclause*batch_size,self.maxvar*batch_size])


    def forward(self, batch_size, biggraph, clause_feat, var_feat, graph_pooler):

        if self.random > 0:
            for r in range(self.random):
                if not self.varvar:
                    r1 = torch.randint(self.random, size=(len(clause_feat),1),device=var_feat.device).float() / self.random
                    clause_feat = torch.cat([clause_feat, r1],1)
                r2 = torch.randint(self.random, size=(len(var_feat), 1),device=var_feat.device).float() / self.random
                var_feat = torch.cat([var_feat, r2],1)

        if self.graph_embedding:
            if not self.varvar:
                clause_hidden_rep = [clause_feat]
            var_hidden_rep = [var_feat]

        if self.varvar:
            h_clause = None
        else:
            h_clause = clause_feat
        h_var = var_feat



        if self.varvar:
            h = h_var
        else:
            h = torch.cat([h_clause,h_var])
        h= self.embedder(h)
        if self.varvar:
            h_var = h
        else:
            h_clause, h_var = torch.split(h,[self.maxclause*batch_size,self.maxvar*batch_size])


        for layer in range(self.num_layers-1):
            h_clause, h_var  = self.next_layer_eps(h_clause,h_var, layer, biggraph,batch_size)
            if self.graph_embedding:
                if not self.varvar:
                    clause_hidden_rep.append(h_clause)
                var_hidden_rep.append(h_var)

        # add last fully adjacent layer see https://arxiv.org/abs/2006.05205
        if self.lfa:
            batched_var = torch.reshape(h_var,(batch_size, self.maxvar, -1))
            lin_var = self.var_lfa(torch.transpose(batched_var,1,2))
            h_var = torch.reshape(torch.transpose(lin_var, 1,2),(batch_size*self.maxvar,-1))

            if not self.varvar:
                batched_clause = torch.reshape(h_var,(batch_size, self.maxclause, -1))
                lin_clause = self.clause_lfa(torch.transpose(batched_clause,1,2))
                h_clause = torch.reshape(torch.transpose(lin_clause, 1,2),(batch_size*self.maxclause,-1))
            if self.graph_embedding:
                if not self.varvar:
                    clause_hidden_rep.append(h_clause)
                var_hidden_rep.append(h_var)

        #below reshape to usual batch form, ie batch index is first dim
        if self.var_classification:
            if self.clause_classification:
                return self.fc1(torch.cat([torch.reshape(h_clause, (batch_size, -1, self.output_dim)),
                                           torch.reshape(h_var,(batch_size, -1, self.output_dim))],axis=1))
            return torch.reshape(self.fc1(h_var), (batch_size, -1, self.output_dim))
        if self.clause_classification:
            return torch.reshape(self.fc1(h_clause), (batch_size, -1, self.output_dim))

        #below graph_embedding only
        score_over_layer = 0

        #perform pooling over all nodes in each graph in every layer
        if self.varvar:
            for layer, h_var in enumerate(var_hidden_rep):
                pooled_h = torch.sparse.mm(graph_pooler, h_var)
                #below this can be considered as  page rank, see https://arxiv.org/abs/2006.07988
                score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)
        else:
            for layer, (h_clause,h_var) in enumerate(zip(clause_hidden_rep,var_hidden_rep)):
                h = torch.cat([h_clause, h_var])
                pooled_h = torch.sparse.mm(graph_pooler, h)
                #below this can be considered as  page rank, see https://arxiv.org/abs/2006.07988
                score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        return score_over_layer


def main():

    varvar = False
    model = GraphCNNSAT(var_classification=True, clause_classification=False, graph_embedding = False, mPGSO=True, maxclause = 20000000, maxvar = 5000000, lfa = False, graph_norm = True, num_layers=5, graph_type="clause.var")
    model.float()
    tds = GraphDataset('../data/test_arup/1.graph', cachesize=0, path_prefix="/home/infantes/code/sat/data/")
    m,labels = tds.getitem(0)
    batch_graph=[m]

    batch_size = len(batch_graph)

    clause_feat, var_feat, nclause, nvar = get_feat(batch_graph, varvar, model.maxclause, model.maxvar, dtype=torch.float)

    device=torch.device("cuda:0")
    cpu=torch.device("cpu")
    clause_feat = clause_feat.to(device)
    var_feat = var_feat.to(device)

    #TODO : precompute degrees
    biggraph = big_tensor_from_batch_graph(batch_graph,varvar, model.maxclause,model.maxvar,neg_as_link=True).to(torch.float).to(device)

    model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    start  = time.process_time()
    p = model.forward(batch_size, biggraph, clause_feat, var_feat, None)

    print("forward time " +str(time.process_time()-start))
    loss_fn = nn.L1Loss()
    loss = loss_fn(p,torch.zeros_like(p))

    start  = time.process_time()
    loss.backward()
    print("backward time " +str(time.process_time()-start))
    opt.step()
    opt.zero_grad()

if __name__ == '__main__':
    main()
