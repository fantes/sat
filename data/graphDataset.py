import torch
import scipy.sparse
import random
import numpy as np
import preprocess
import time

class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, filename, permute_vars = False, permute_clauses = False, neg_clauses = True, self_supervised = False, cachesize=100):
        self.fname = filename
        self.cachesize = cachesize
        self.cache = {}
        self.data = []
        self.self_supervised = self_supervised
        f = open(self.fname,'r')
        l = f.readline().strip().split()
        print(l)
        self.nlabels = int(l[0])
        self.maxvar = int(l[1])
        self.nvar = int(l[3])
        self.nclause = int(l[2])
        self.permute_vars = permute_vars
        self.permute_clauses = permute_clauses
        self.neg_clauses = neg_clauses
        if neg_clauses:
            self.nclause = self.nclause+self.nvar
        ll = f.readline()
        while ll:
            l = ll.strip().split()
            labels = l[:self.nlabels]
            graphfiles = l[self.nlabels:]
            self.data.append({'l':labels,'f':graphfiles})
            ll = f.readline()

        if self.permute_vars:
            self.varPermRows = list(range(self.maxvar*2))
            self.varData = [True]*(self.maxvar*2)

        if self.permute_clauses:
            self.clausesPerRows = list(range(self.nclause))
            self.varClause = [True]*self.nclause

        if self.neg_clauses:
            vlist = [v for v in range(0,self.nvar)]
            vneglist = [self.maxvar+v for v in range(0,self.nvar)]
            negclauselist = list(range(0,self.nvar))
            varClause = [True] *2*self.nvar
            self.negClauseMatrix = scipy.sparse.csr_matrix((varClause,(negclauselist*2, vlist + vneglist)),shape=(self.nvar, self.maxvar*2),dtype=bool)


    def __len(self):
        return len(self.data)

    def mpermute(self,ssm, labels,permute_vars=True, permute_clauses=True):
        res = ssm
        if permute_vars:
            varPermuted = np.random.permutation(list(range(self.maxvar*2)))
            vperMatrix = scipy.sparse.csr_matrix((self.varData,(self.varPermRows,varPermuted)),shape=(self.maxvar*2,self.maxvar*2),dtype=bool)
            res =  ssm * vperMatrix


        if permute_clauses:
            clausesPermuted = np.random.permutation(list(range(self.nclause)))
            clausePerMatrix = scipy.sparse.csr_matrix((self.varClause,(self.clausesPerRows,clausesPermuted)),shape=(self.nclause,self.nclause),dtype=bool)
            res = clausePerMatrix *res

        # do not forget to permute labels, if relevant
        return res, labels

    def addNegClauses(self,ssm):
        return scipy.sparse.vstack([self.negClauseMatrix,ssm])


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
                ssm = ssm + newssm

        if len(self.cache) >= self.cachesize and self.cachesize> 0:
            del self.cache[random.choice(list(self.cache.keys()))]
        self.cache[idx] = {'labels':self.data[idx]['l'],'graph':ssm}
        if self.neg_clauses:
            ssm = self.addNegClauses(ssm)
        return self.mpermute(ssm, self.data[idx]['l'],self.permute_vars, self.permute_clauses)

    def getitem(self,idx):
        return self.__getitem__(idx)

def getDataLoader(filename, batch_size, num_workers=10, cachesize=100):
    dset = GraphDataset(filename, cachesize=cachesize)
    loader = torch.utils.data.DataLoader(dset, batch_size= batch_size, shuffle=True, num_workers=num_workers)
    return loader


def main():
    start = time.process_time()
    preprocess.preprocess('/data1/infantes/systerel/T102.2.1.cnf', './test', 1)
    end = time.process_time()
    print("preprocess time: " + str(end-start))
    start = time.process_time()
    tds = GraphDataset('./test/T102.2.1.graph', neg_clauses = True,  self_supervised = False, cachesize=0)
    end = time.process_time()
    print("init time: " + str(end-start))
    start = time.process_time()
    ssm , labels = tds.getitem(0)
    end = time.process_time()
    print("get item time: " + str(end-start))
    print(ssm.shape)

    start = time.process_time()
    varArity = np.asarray(ssm.sum(axis=0)).flatten()
    end =  time.process_time()
    print('varArity compute time: ' + str(end-start))

    start = time.process_time()
    clauseArity = np.asarray(ssm.sum(axis=1)).flatten()
    end =  time.process_time()
    print('clauseArity compute time: ' + str(end-start))






if __name__ == '__main__':
    main()
