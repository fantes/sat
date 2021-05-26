import torch
import scipy.parse
import random
import numpy as np

class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, filename, permute = True, self_supervised = False, cachesize=100):
        self.fname = filename
        self.cachesize = cachesize
        self.cache = {}
        self.data = []
        self.self_supervised = self_supervised
        f = open(self.fname,'r')
        l = f.readline().strip().split()
        self.nlabels = int(l[0])
        self.maxvar = int(l[1])
        self.maxclause = int(l[2])
        self.permute = permute
        ll = f.readline()
        while ll:
            l = f.readline().strip().split()
            labels = l[:self.nlabels]
            graphfiless = l[self.nlabels:]
            self.data.append({'l'=labels,'f'=graphfiles])
            ll = f.readline()

    def __len(self):
        return len(self.data)

    def mpermute(ssm, labels):
        varPermuted = np.random.permutation(list(range(self.maxvar*2+1)))
        varPermRows = list(range(self.maxvar*2+1))
        varData = [True]*(self.maxvar*2+1)
        vperMatrix = scipy.sparse.coo_matrix((varData,(varPerRows,varPermuted)),shape=(2*maxvar+1,2*maxvar+1),dtype=bool)
        varPermuted = vperMatrix * ssm

        clausesPermuted = np.random.permutation(list(range(self.maxclause)))
        clausesPerRows = list(range(self.maxclause))
        varClause = [True]*self.maxclause
        clausePerMatrix = scipy.sparse.coo_matrix((varClause,(clausesPerRows,clausesPermuted)),shape=(mxclause,maxclause),dtype=bool)
        clausesPermuted = varPermuted * clausePerMatrix

        # do not forget to permute labels, if relevant
        return clausesPermuted, labels

    def addNegClauses(ssm):



    def __getitem__(self, idx):
        if self.cache.has_key(idx):
            if self.permute:
                return mpermute(self.cache[idx]['graph'],self.cache[idx]['labels'])
            else:
                return self.cache[idx]['graph'],self.cache[idx]['labels']
        ssm = scipy.sparse.coo_matrix((2*maxvar+1,maxclause),dtype=bool)
        for graphfile in self.data[idx]['f']:
            ssm = ssm + scipy.sparse.load_npz(graphfile)
        if len(self.cache) >= self.cachesize:
            del self.cache[random.choice(list(self.cache.keys()))]
        self.cache[idx] = {'labels':self.data[idx]['l'],'graph'=ssm}
        if self.permute:
            return mpermute(ssm, self.data[idx]['l'])
        else:
            return ssm, self.data[idx]['l']

def getDataLoader(filename, batch_size, num_workers=10, cachesize=100):
    dset = GraphDataset(filename, cachesize=cachesize)
    loader = torch.utils.data.DataLoader(dset, 'batch_size'= batch_size, 'shuffle'=True, 'num_workers'=num_workers)
    return loader
