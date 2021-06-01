import torch
import scipy.sparse
import numpy as np




def big_tensor_from_batch_graph(graphs, maxclause, maxvar,half_compute=True):
    #batch graph is maxclause*batch_size x maxvar*batch_size
    for g in graphs:
        g.resize(maxclause,maxvar)
    big_mat = scipy.sparse.block_diag(graphs,format="coo", dtype= np.bool)
    if not half_compute:
        big_mat = scipy.sparse.block_diag([big_mat, big_mat.transpose()])
    big_tensor = torch.sparse_coo_tensor([big_mat.row, big_mat.col], big_mat.data, big_mat.shape, dtype=torch.int32)
    return big_tensor


def build_graph_pooler(batch_size,nclause,nvar, maxclause,maxvar):
    blocks = []
    for i in range(batch_size):
        blocks.append(np.mat(np.ones((1,maxclause+maxvar))))
        if self.graph_pooling_type == "average":
            blocks[-1] = blocks[-1]/(nclause[i] + nvar[i])
    spgraphpooler = scipy.sparse.block_diag(blocks)
    return torch.sparse_coo_tensor([spgraphpooler.row,spgraphpooler.col],spgraphpooler.data, spgraphpooler.shape, dtype=torch.float32).to(self.device)


def get_feat(batch_graph, maxclause, maxvar):
    clause_arities = []
    var_arities = []
    nclause = []
    nvar = []
    batch_size = len(batch_graph)

    for ssm in batch_graph:
        nclause.append(ssm.shape[0])
        nvar.append(ssm.shape[1])
        clause_arities.append(torch.cat([torch.tensor(np.asarray(ssm.sum(axis=1).flatten())[0]),torch.zeros(maxclause-nclause[-1],dtype=torch.int32)]))
        var_arities.append(torch.cat([torch.tensor(np.asarray(ssm.sum(axis=0).flatten())[0]),torch.zeros(maxvar-nvar[-1],dtype=torch.int32)]))

    clause_feat = torch.cat(clause_arities, 0)
    clause_feat.unsqueeze_(1)
    var_feat = torch.cat(var_arities,0)
    var_feat.unsqueeze_(1)

    return clause_feat, var_feat, nclause, nvar


def postproc(data, maxclause,maxvar,half_compute=True, graph_pool=False):
    batch_size = len(data)
    graph_batch=[d['graph'] for d in data]
    label_batch = [d['labels'] for d in data]
    clause_feat, var_feat, nclause, nvar = get_feat(graph_batch, maxclause, maxvar)
    biggraph = big_tensor_from_batch_graph(graph_batch,maxclause,maxvar,half_compute).to(torch.float32)

    if graph_pool:
        graph_pooler = build_graph_pooler(batch_size, nclause, nvar, maxclause, maxvar)
    else:
        graph_pooler = None

    return batch_size, biggraph, clause_feat, var_feat, graph_pooler, label_batch
