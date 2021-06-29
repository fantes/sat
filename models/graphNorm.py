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

    def print_mem(self, name, var):
        print(name+": " + str(var.nelement()*var.element_size()))
        print("shape: " + str(var.shape))

    def forward(self, h):
        batchsize = int(h.shape[0]/(self.maxclause+self.maxvar))
        mean = torch.zeros(batchsize, h.shape[1],dtype=torch.float32).to(h.device)
        batch_index = torch.arange(batchsize).repeat_interleave(self.maxvar+self.maxclause).to(h.device)
        batch_index = batch_index.view((-1,1)).expand_as(h).to(h.device)
        mean = torch.scatter_add(mean, 0, batch_index, h.to(torch.float32))
        mean /=  (self.maxvar+self.maxclause)
        mean = mean.repeat_interleave(self.maxvar+self.maxclause, dim=0)

        mean *= -self.mean_scale.to(h.device)
        mean += h

        std = torch.zeros(batchsize, h.shape[1],dtype=torch.float32).to(h.device)
        std.scatter_add_(0, batch_index, mean.pow(2))
        std = (std/(self.maxclause+self.maxvar) + 1e-6).sqrt()
        std = std.repeat_interleave(self.maxclause+self.maxvar,dim=0)

        ret = self.weight * mean / std + self.bias
        ret = ret.to(h.dtype)
        return ret
