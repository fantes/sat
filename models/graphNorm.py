import torch
import torch.nn as nn

class GraphNorm(nn.Module):
    # graphnorm, see https://arxiv.org/abs/2009.03294
    #https://github.com/lsj2408/GraphNorm

    def __init__(self, hidden_dim, maxclause, maxvar):
        super(GraphNorm, self).__init__()
        self.maxclause = maxclause
        self.maxvar = maxvar
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias =  nn.Parameter(torch.zeros(hidden_dim))
        self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, h):
        batchsize = int(h.shape[0]/(self.maxclause+self.maxvar))
        mean = torch.zeros(batchsize, h.shape[1]).to(h.device)
        batch_index = torch.arange(batchsize).to(h.device).repeat_interleave(self.maxvar+self.maxclause)
        batch_index = batch_index.view((-1,1)).expand_as(h)
        mean.scatter_add_(0, batch_index, h)
        mean = mean / (self.maxvar+self.maxclause)
        mean = mean.repeat_interleave(self.maxvar+self.maxclause, dim=0)

        sub = h - mean * self.mean_scale

        std = torch.zeros(batchsize, h.shape[1]).to(h.device)
        std.scatter_add_(0, batch_index, sub.pow(2))
        std = (std/(self.maxclause+self.maxvar) + 1e-6).sqrt()
        std = std.repeat_interleave(self.maxclause+self.maxvar,dim=0)

        return self.weight * sub / std + self.bias
