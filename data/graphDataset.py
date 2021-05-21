import torch
import scipy.parse
import random

class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, filename,cachesize=100):
        self.fname = filename
        self.cachesize = cachesize
        self.cache = {}
        self.data = []
        f = open(self.fname,'r')
        l = f.readline().strip().split()
        self.nlabels = l[0]
        self.maxvar = l[1]
        self.maxclause = l[2]
        ll = f.readline()
        while ll:
            l = f.readline().strip().split()
            labels = l[:self.nlabels]
            graphfiless = l[self.nlabels:]
            self.data.append({'l'=labels,'f'=graphfiles])
            ll = f.readline()

    def __len(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cache.has_key(idx):
            return self.cache[idx]
        ssm = scipy.sparse.coo_matrix((maxvar+maxclause,maxvar+maxclause),dtype=bool)
        for graphfile in self.data[idx]['f']:
            ssm = ssm + scipy.sparse.load_npz(graphfile)
        if len(self.cache) >= self.cachesize:
            del self.cache[random.choice(list(self.cache.keys()))]
        self.cache[idx] = {'labels':self.data[idx]['l'],'graph'=ssm}
        return ssm, self.data[idx]['l']

def getDataLoader(filename, batch_size, num_workers=10, cachesize=100):
    dset = GraphDataset(filename, cachesize=cachesize)
    loader = torch.utils.data.DataLoader(dset, 'batch_size'= batch_size, 'shuffle'=True, 'num_workers'=num_workers)
    return loader
