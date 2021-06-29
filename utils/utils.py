import torch
import scipy.sparse
import numpy as np

def conv_vindex(v, nvar, neg_as_link):
    if neg_as_link:
        if v<0:
            return -v-1
        return v-1
    else:
        if v<0:
            return -v+nvar-1
        return v-1

def convert_labels(labels, neg_as_link, maxvar, device):
    target = torch.zeros((len(labels),maxvar), dtype=torch.long, device=device)
    for count, b in enumerate(labels):
        # batch
        for l in b:
            #labels in batch
            if l < 0:
                target[count, conv_vindex(l, maxvar, neg_as_link)] = 1
            else:
                target[count, conv_vindex(l,maxvar,neg_as_link)] = 2
    return target





def big_tensor_from_batch_graph(graphs, varvar, maxclause, maxvar, neg_as_link):
    #batch graph is maxclause*batch_size x maxvar*batch_size
    for g in graphs:
        if varvar:
            g.resize(maxvar,maxvar)
        else:
            g.resize(maxclause,maxvar)
    big_mat = graphs[0].tocoo(copy=False)
    for g in graphs[1:]:
        if neg_as_link:
            big_mat = scipy.sparse.bmat([[big_mat,None],[None,g.tocoo(copy=False)]],format="coo",dtype=np.short)
        else:
            big_mat = scipy.sparse.bmat([[big_mat,None],[None,g.tocoo(copy=False)]],format="coo",dtype=np.short)
    if neg_as_link:
        big_tensor = torch.sparse_coo_tensor([big_mat.row, big_mat.col], big_mat.data, big_mat.shape, dtype=torch.short)
    else:
        big_tensor = torch.sparse_coo_tensor([big_mat.row, big_mat.col], big_mat.data, big_mat.shape, dtype=torch.short)
    return big_tensor


def build_graph_pooler(batch_size,varvar,nclause,nvar, maxclause,maxvar):
    blocks = []
    for i in range(batch_size):
        qblocks.append(np.mat(np.ones((1,maxclause+maxvar))))
        if self.graph_pooling_type == "average":
            blocks[-1] = blocks[-1]/(nclause[i] + nvar[i])
    spgraphpooler = scipy.sparse.block_diag(blocks)
    return torch.sparse_coo_tensor([spgraphpooler.row,spgraphpooler.col],spgraphpooler.data, spgraphpooler.shape, dtype=torch.float32).to(self.device)


def get_feat(batch_graph, varvar, maxclause, maxvar,dtype=torch.float):
    clause_arities = []
    var_arities = []
    nclause = []
    nvar = []
    batch_size = len(batch_graph)

    for ssm in batch_graph:
        if not varvar: #ie clause x var
            nclause.append(ssm.shape[0])
            clause_arities.append(torch.cat([torch.tensor(np.asarray(np.absolute(ssm).sum(axis=1).flatten())[0]),torch.zeros(maxclause-nclause[-1],dtype=dtype)]))
        nvar.append(ssm.shape[1])
        ar = torch.tensor(np.asarray(ssm.sum(axis=0).flatten())[0])
        if varvar: # we store on disk only directed links
            ar += torch.tensor(np.asarray(np.absolute(ssm).sum(axis=1).flatten())[0])
        var_arities.append(torch.cat([ar,torch.zeros(maxvar-nvar[-1],dtype=dtype)]))

    if not varvar:
        clause_feat = torch.cat(clause_arities, 0)
        clause_feat.unsqueeze_(1)

    var_feat = torch.cat(var_arities,0)
    var_feat.unsqueeze_(1)

    if not varvar:
        return clause_feat, var_feat, nclause, nvar
    return None, var_feat, None, nvar


def postproc(data, maxclause,maxvar,varvar, graph_pool=False, neg_as_link = True):
    batch_size = len(data)
    graph_batch=[d[0] for d in data]
    label_batch = [d[1] for d in data]
    clause_feat, var_feat, nclause, nvar = get_feat(graph_batch, varvar, maxclause, maxvar)
    biggraph = big_tensor_from_batch_graph(graph_batch,varvar,maxclause,maxvar,neg_as_link)

    if graph_pool:
        graph_pooler = build_graph_pooler(batch_size, nclause, nvar, maxclause, maxvar)
    else:
        graph_pooler = None


    return batch_size, biggraph, clause_feat, var_feat, graph_pooler, label_batch
