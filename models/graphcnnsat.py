from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("models/")
sys.path.append("../data/")
sys.path.append("../utils/")
from utils import *
from graphDataset import GraphDataset
from graphNorm import GraphNorm
from mlp import MLP

import numpy as np
import scipy.sparse

class GraphCNNSAT(nn.Module):

    def __init__(self, num_layers=10, num_mlp_layers=2,  hidden_dim=1, output_dim=2, final_dropout=0.5, random = 0, maxclause = 10, maxvar = 20, graph_type = "var.var", var_classification = True, clause_classification = False, graph_embedding = False, PGSO=False, mPGSO=False, graph_norm = True, neighbor_pooling_type = "sum", graph_pooling_type = "average", lfa = True,  device=torch.device("cuda:1")):

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
        self.norms = torch.nn.ModuleList()

        self.num_mlp_layers = num_mlp_layers

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(self.num_mlp_layers, self.input_dim, self.hidden_dim, self.hidden_dim,self.graph_norm,self.maxclause,self.maxvar))
            else:
                self.mlps.append(MLP(self.num_mlp_layers, self.hidden_dim, self.hidden_dim, self.hidden_dim,self.graph_norm,self.maxclause,self.maxvar))

            if self.graph_norm:
                self.norms.append(GraphNorm(self.hidden_dim,self.maxclause, self.maxvar))
            else:
                self.norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.mlps.to(self.device)
        self.norms.to(self.device)

        self.var_classification = var_classification

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
            if not varvar:
                self.clause_lfa = nn.Linear(self.maxclause, self.maxclause).to(device)


    def print_mem(self, name, var):
        print(name+": " + str(var.nelement()*var.element_size()))
        print("shape: " + str(var.shape))

    def next_layer_eps(self, h_clause, h_var, layer, biggraph,batch_size):
        # biggraph is (maxclause * batchsize ) x (mavvar * batchsize)
        # h should be (maxvar+maxclause)*batch_size

        # PGSO see https://arxiv.org/abs/2101.10050 (page 5)

        if self.neighbor_pooling_type == "average" or self.PGSO:
            if self.varvar:
                degrees = torch.sparse.mm(biggraph, torch.ones((biggraph.shape[0], 1)).to(self.device)).to(torch.short)
            else:
                tempones = torch.ones((biggraph.shape[1], 1)).to(self.device)
                degree_clauses = torch.sparse.mm(biggraph, tempones).to(torch.short)
                tempones = torch.ones((biggraph.shape[0], 1)).to(self.device)
                degree_vars = torch.sparse.mm(torch.transpose(biggraph,0,1), tempones).to(torch.short)
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


        if not self.varvar: #ie graph is clause x var

            if self.PGSO:
                dje3 = torch.pow(degree_vars.to_dense() * la,le3)
                h_var_dje3 = torch.mul(h_var,dje3)
                clause_pooled = torch.sparse.mm(biggraph, h_var_dje3)
                die2 = torch.pow(degree_clauses,le2)
                clause_pooled = torch.mul(clause_pooled, die2)

                dje3 = torch.pow(degree_clauses,le3)
                h_clause_dje3 = torch.mul(h_clause,dje3)
                var_pooled = torch.sparse.mm(torch.transpose(biggraph,0,1), h_clause_dje3)
                die2 = torch.pow(degree_vars * la ,le2)
                var_pooled = torch.mul(var_pooled, die2)

            else:
                clause_pooled = torch.sparse.mm(biggraph, h_var.to(torch.float)).to(torch.half)
                var_pooled = torch.sparse.mm(torch.transpose(biggraph,0,1), h_clause.to(torch.float)).to(torch.half)



            if self.neighbor_pooling_type == "average": #should be subsumed by PGSO
                clause_pooled = clause_pooled/degree_clauses.expand_as(clause_pooled)
                var_pooled = var_pooled/degree_vars.expand_as(var_pooled)

            pooled = torch.cat([clause_pooled, var_pooled])
        else:
            h = h_var

            if self.PGSO:
                dje3 = torch.pow(degrees * la, le3)
                h_dje3 = torch.mul(h,dje3)
                pooled = torch.sparse.mm(biggraph, h_dje3)
                die2 = torch.pow(degrees, le2)
                pooled = torch.mul(pooled, die2)
            else:
                pooled = torch.sparse.mm(biggraph, h)

            if self.neighbor_pooling_type == "average": #should be subsumed by graph_shift
                pooled = pooled/degree.expand_as(pooled)


        if self.varvar:
            h = h_var
        else:
            h = torch.cat([h_clause, h_var])

        if self.PGSO:
            dai_e1 = torch.pow(degrees, le1)
            pooled = (dai_e1 * lm1 * h) + lm3 * h + lm2 * pooled
        else:
            pooled +=  (1+self.eps[layer])*h
        pooled = F.relu(self.norms[layer](self.mlps[layer](pooled)))

        if self.varvar:
            return None, pooled
        else:
            return torch.split(pooled,[self.maxclause*batch_size,self.maxvar*batch_size])


    def forward(self, batch_size, biggraph, clause_feat, var_feat, graph_pooler):

        if self.random > 0:
            for r in range(self.random):
                if not self.varvar:
                    r1 = torch.randint(self.random, size=(len(clause_feat),1)).half() / self.random
                    clause_feat = torch.cat([clause_feat, r1],1).to(self.device)
                r2 = torch.randint(self.random, size=(len(var_feat), 1)).half() / self.random
                var_feat = torch.cat([var_feat, r2],1).to(self.device)

        if self.graph_embedding:
            if not self.varvar:
                clause_hidden_rep = [clause_feat]
            var_hidden_rep = [var_feat]

        if self.varvar:
            h_clause = None
        else:
            h_clause = clause_feat.to(self.device)
        h_var = var_feat.to(self.device)


        for layer in range(self.num_layers-1):
            print(layer)
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



        if self.var_classification:
            if self.clause_classification:
                return torch.softmax(self.fc1(torch.cat([h_clause,h_var],axis=0)), 1)
            return torch.softmax(self.fc1(h_var), 1)
        if self.clause_classification:
            return torch.softmax(self.fc1(h_clause), 1)


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
    model = GraphCNNSAT(var_classification=True, clause_classification=False, graph_embedding = False, mPGSO=False, maxclause = 20000000, maxvar = 9000000, lfa = False, graph_norm = True, num_layers=4, graph_type="clause.var")
    model.half()
    tds = GraphDataset('../data/test_clausevar/T102.2.1.graph', cachesize=0, path_prefix="/home/infantes/code/sat/data/", neg_clauses=True, varvar=varvar)
    m,labels = tds.getitem(0)
    batch_graph=[m]

    batch_size = len(batch_graph)



    clause_feat, var_feat, nclause, nvar = get_feat(batch_graph, varvar, model.maxclause, model.maxvar, dtype=torch.half)

    #TODO : precompute degrees
    biggraph = big_tensor_from_batch_graph(batch_graph,varvar, model.maxclause,model.maxvar).to(model.device).to(torch.float)



    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    start  = time.process_time()

    p = model.forward(batch_size, biggraph, clause_feat, var_feat, None)
    print("forward time " +str(time.process_time()-start))
    loss_fn = nn.L1Loss()
    loss = loss_fn(p,torch.zeros_like(p))

    loss.backward()
    opt.step()
    opt.zero_grad()

if __name__ == '__main__':
    main()
