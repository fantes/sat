import torch
import scipy.sparse
import random
import numpy as np
import time
import sys
sys.path.append("../utils/")
from utils.utils import *



class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, filename, permute_vars = False, permute_clauses = False, neg_clauses = True, self_supervised = False, cachesize=100, path_prefix="./"):
        self.fname = filename
        self.cachesize = cachesize
        self.cache = {}
        self.data = []
        self.path_prefix = path_prefix
        self.self_supervised = self_supervised
        f = open(self.fname,'r')
        l = f.readline().strip().split()
        self.nsamples = int(l[0])
        self.nclause = int(l[1])
        self.nvar = int(l[2])
        cvt = l[3]
        if cvt == "clause.var":
            self.varvar = False
        elif cvt == "var.var":
            self.varvar = True
        else:
            raise RuntimeError("unknown graph type " + cvt)

        nlt = l[4]
        if nlt == "neg_as_link":
            self.neg_as_link = True
        elif nlt == "neg_as_var":
            self.neg_as_link = False
        else:
            raise RuntimeError("unknown neg type " + nlt)

        self.permute_vars = permute_vars
        self.permute_clauses = not self.varvar and permute_clauses
        self.neg_clauses = not self.neg_as_link and  neg_clauses
        if neg_clauses:
            self.nclause = self.nclause+self.nvar
        ll = f.readline()
        while ll:
            l = ll.strip().split()
            nlabels = int(l[0])
            labels = [int(v) for v in l[1:nlabels+1]]
            graphfiles = l[nlabels+1:]
            self.data.append({'labels':labels,'f':graphfiles})
            ll = f.readline()

        if self.permute_vars:
            self.varPermRows = list(range(self.nvar*2))
            self.varData = [True]*(self.nvar*2)

        if self.permute_clauses:
            self.clausesPerRows = list(range(self.nclause))
            self.varClause = [True]*self.nclause

        if self.neg_clauses:
            if self.varvar:
                vlist = [v for v in range(0,self.nvar)]
                vneglist = [self.nvar+v for v in range(0,self.nvar)]
                varClause = [True] *self.nvar
                self.negClauseMatrix = scipy.sparse.csr_matrix((varClause,(vlist, vneglist)),shape=(self.nvar*2, self.nvar*2),dtype=bool)
            else:
                vlist = [v for v in range(0,self.nvar)]
                vneglist = [self.nvar+v for v in range(0,self.nvar)]
                negclauselist = list(range(0,self.nvar))
                varClause = [True] *2*self.nvar
                self.negClauseMatrix = scipy.sparse.csr_matrix((varClause,(negclauselist*2, vlist + vneglist)),shape=(self.nvar, self.nvar*2),dtype=bool)


    def __len__(self):
        return len(self.data)

    def mpermute(self,ssm, labels):
        res = ssm
        if self.permute_vars:
            if self.neg_as_link:
                varPermuted = np.random.permutation(list(range(self.nvar)))
            else:
                varPermuted = np.random.permutation(list(range(self.nvar*2)))
            if not self.varvar:
                if self.neg_as_link:
                    vperMatrix = scipy.sparse.csr_matrix((self.varData,(self.varPermRows,varPermuted)),shape=(self.nvar,self.nvar),dtype=byte)
                else:
                    vperMatrix = scipy.sparse.csr_matrix((self.varData,(self.varPermRows,varPermuted)),shape=(self.nvar*2,self.nvar*2),dtype=bool)
                res =  ssm * vperMatrix
            else :
                vperMatrix = scipy.sparse.csr_matrix((self.varData,(self.varPermRows,varPermuted)),shape=(self.nvar*2,self.nvar*2),dtype=bool)
                res = vperMatrix * ssm * vperMatrix

            labels = []
            if self.neg_as_link:
                for l in labels:
                    if l < 0:
                        labels.append(-varPermuted[-l])
                    else:
                        labels.append(varPermuted[l])
            else:
                l.append(varPermuted[conv_vindex(l,self.nvar,False)])


        if self.permute_clauses:
            clausesPermuted = np.random.permutation(list(range(self.nclause)))
            clausePerMatrix = scipy.sparse.csr_matrix((self.varClause,(self.clausesPerRows,clausesPermuted)),shape=(self.nclause,self.nclause),dtype=bool)
            res = clausePerMatrix *res

        return res, labels

    def addNegClauses(self,ssm):
        if self.neg_as_link:
            return ssm
        if self.varvar:
            return ssm+self.negClauseMatrix
        else:
            return scipy.sparse.vstack([self.negClauseMatrix,ssm])


    def __getitem__(self, idx):
        if idx in self.cache:
            ssm, labels = self.mpermute(self.cache[idx][0], self.cache[idx][1])
            return (ssm, labels)
        ssm = None
        for graphfile in self.data[idx]['f']:
            newssm = scipy.sparse.load_npz(self.path_prefix+graphfile).tocsr()
            #below: allow to read clause x var files to give var var matrices
            if self.nclause != 0 and self.varvar:
                newssm = newssm.transpose() * newssm
            if ssm is None:
                ssm = newssm
            else:
                if self.varvar:
                    ssm += newssm
                else:
                    ssm = scipy.sparse.vstack([ssm,newssm])

        if len(self.cache) >= self.cachesize and self.cachesize> 0:
            del self.cache[random.choice(list(self.cache.keys()))]
        if self.cachesize>0:
            self.cache[idx] = (ssm, self.data[idx]['labels'])
        if self.neg_clauses:
            ssm = self.addNegClauses(ssm)
        ssm,labels =  self.mpermute(ssm, self.data[idx]['labels'])
        return ssm, labels

    def getitem(self,idx):
        return self.__getitem__(idx)


    def getDataLoader(self, batch_size,  maxclause, maxvar, varvar = True, graph_pool = False, num_workers=1):
        #dset = GraphDataset(filename, cachesize=cachesize)
        loader = torch.utils.data.DataLoader(self, batch_size= batch_size, shuffle=True, num_workers=num_workers,pin_memory=False, collate_fn = lambda x : postproc(x, maxclause, maxvar, self.varvar, graph_pool, self.neg_as_link))
        return loader


def main():
    # start = time.process_time()
    # preprocess.preprocess('/data1/infantes/systerel/T102.2.1.cnf', './test', 10)
    # end = time.process_time()
    # print("preprocess time: " + str(end-start))
    start = time.process_time()
    tds = GraphDataset('./test_arup/1.graph',  self_supervised = False, cachesize=0)
    end = time.process_time()
    print("init time: " + str(end-start))
    start = time.process_time()
    ssm, labels = tds.getitem(0)
    print("labels")
    print(labels)
    end = time.process_time()
    print("get item time: " + str(end-start))
    print(ssm.shape)

    ssm, labels = tds.getitem(1)
    print("labels")
    print(labels)
    end = time.process_time()
    print("get item time: " + str(end-start))
    print(ssm.shape)

    ssm, labels = tds.getitem(14)
    print("labels")
    print(labels)
    end = time.process_time()
    print("get item time: " + str(end-start))
    print(ssm.shape)


    # start = time.process_time()
    # varArity = np.asarray(ssm.sum(axis=0)).flatten()
    # end =  time.process_time()
    # print('varArity compute time: ' + str(end-start))

    # start = time.process_time()
    # clauseArity = np.asarray(ssm.sum(axis=1)).flatten()
    # end =  time.process_time()
    # print('clauseArity compute time: ' + str(end-start))

    # dl = tds.getDataLoader(2, 20000000, 5000000)
    # for i_batch, data in enumerate(dl):
    #     print(data)




if __name__ == '__main__':
    main()
