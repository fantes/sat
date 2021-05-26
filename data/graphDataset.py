import torch
import scipy.sparse
import random
import numpy as np
import preprocess

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
        self.nvar = int(l[2])
        self.nclause = int(l[3])
        self.permute = permute
        ll = f.readline()
        while ll:
            l = ll.strip().split()
            labels = l[:self.nlabels]
            graphfiles = l[self.nlabels:]
            self.data.append({'l':labels,'f':graphfiles})
            ll = f.readline()

    def __len(self):
        return len(self.data)

    def mpermute(ssm, labels):
        varPermuted = np.random.permutation(list(range(self.maxvar*2+1)))
        varPermRows = list(range(self.maxvar*2+1))
        varData = [True]*(self.maxvar*2+1)
        vperMatrix = scipy.sparse.coo_matrix((varData,(varPerRows,varPermuted)),shape=(maxvar,maxvar),dtype=bool)
        varPermuted =  ssm * vperMatrix

        clausesPermuted = np.random.permutation(list(range(self.maxclause)))
        clausesPerRows = list(range(self.maxclause))
        varClause = [True]*self.maxclause
        clausePerMatrix = scipy.sparse.coo_matrix((varClause,(clausesPerRows,clausesPermuted)),shape=(mxclause,maxclause),dtype=bool)
        clausesPermuted = clausePerMatrix *varPermuted

        # do not forget to permute labels, if relevant
        return clausesPermuted, labels

    def addNegClauses(self,ssm):
        vlist = [v for v in range(1,self.nvar+1)]
        vneglist = [self.maxvar/2+v for v in range(1,self.nvar+1)]
        negclauselist = list(range(0,self.nvar))
        varClause = [True] *2*self.nvar
        negClauseMatrix = scipy.sparse.coo_matrix((varClause,(negclauselist*2, vlist + vneglist)),shape=(self.nvar, self.maxvar),dtype=bool)
        return scipy.sparse.vstack([negClauseMatrix,ssm])


    def __getitem__(self, idx):
        if idx in self.cache:
            if self.permute:
                return mpermute(self.cache[idx]['graph'],self.cache[idx]['labels'])
            else:
                return self.cache[idx]['graph'],self.cache[idx]['labels']
        ssm = None
        for graphfile in self.data[idx]['f']:
            newssm = scipy.sparse.load_npz(graphfile)
            if ssm is None:
                ssm = newssm
            else:
                ssm = scipy.sparse.vstack([ssm,newssm])

        if len(self.cache) >= self.cachesize and self.cachesize> 0:
            del self.cache[random.choice(list(self.cache.keys()))]
        self.cache[idx] = {'labels':self.data[idx]['l'],'graph':ssm}
        if self.permute:
            return mpermute(ssm, self.data[idx]['l'])
        else:
            return ssm, self.data[idx]['l']

    def getitem(self,idx):
        return self.__getitem__(idx)

def getDataLoader(filename, batch_size, num_workers=10, cachesize=100):
    dset = GraphDataset(filename, cachesize=cachesize)
    loader = torch.utils.data.DataLoader(dset, batch_size= batch_size, shuffle=True, num_workers=num_workers)
    return loader


def main():
    preprocess.preprocess('/data1/infantes/systerel/ex.cnf', './test', 2, maxvar=5)
    tds = GraphDataset('./test/ex.graph', permute=False, self_supervised = False, cachesize=0)
    ssm , labels = tds.getitem(0)
    print(ssm.toarray())
    print(tds.addNegClauses(ssm).toarray())

if __name__ == '__main__':
    main()
