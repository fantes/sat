import torch
import scipy.sparse
import random
import numpy as np
import time
import sys
sys.path.append("../")
from utils.utils import *



class GraphDataset(torch.utils.data.Dataset):

    def __init__(self, filenames, maxvar, permute_vars = False, permute_clauses = False, neg_clauses = True, self_supervised = False, normalize = False, cachesize=100, path_prefix="./"):
        self.fnames = filenames
        self.cachesize = cachesize
        self.cache = {}
        self.data = []
        self.normalize = normalize
        self.path_prefix = path_prefix
        self.self_supervised = self_supervised
        self.nsamples = []
        self.nclause = []
        self.nvar = []
        self.maxvar = maxvar
        self.totalNLabels = 0
        self.negClauseMatrix = []
        findex = 0

        for findex, filename in enumerate(filenames):
            f = open(filename,'r')
            print("opening " + filename)
            l = f.readline().strip().split()
            self.nsamples.append(int(l[0]))
            self.nclause.append(int(l[1]))
            self.nvar.append(int(l[2]))

            #TODO : below check consistency
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
                self.nclause[findex] += self.nvar[findex]
            ll = f.readline()

            while ll:
                l = ll.strip().split()
                nlabels = int(l[0])
                labels = [int(v) for v in l[1:nlabels+1]]
                self.totalNLabels += len(labels)
                graphfiles = l[nlabels+1:]
                self.data.append({'labels':labels,'f':graphfiles,'id':findex})
                ll = f.readline()

            if self.permute_vars:

                if self.neg_as_link:
                    self.varPermRows = list(range(self.maxvar))
                    self.varData = [1]*(self.maxvar)
                else:
                    self.varPermRows = list(range(self.maxvar*2))
                    self.varData = [1]*(self.maxvar*2)

            if self.permute_clauses:
                self.clausesPerRows.append(list(range(self.nclause[findex])))
                if self.neg_as_link:
                    self.varClause.append([1]*self.nclause[findex])
                else:
                    self.varClause.append([1]*self.nclause[findex])

            if self.neg_clauses:
                if self.varvar:
                    vlist = [v for v in range(0,self.nvar)]
                    vneglist = [self.maxvar+v for v in range(0,self.nvar[findex])]
                    varClause = [True] *self.nvar[findex]
                    self.negClauseMatrix.append(scipy.sparse.csr_matrix((varClause,(vlist, vneglist)),shape=(self.maxvar*2, self.maxvar*2),dtype=bool))
                else:
                    vlist = [v for v in range(0,self.nvar[findex])]
                    vneglist = [self.maxvar+v for v in range(0,self.nvar[findex])]
                    negclauselist = list(range(0,self.nvar[findex]))
                    varClause = [1] *2*self.nvar[findex]
                    self.negClauseMatrix.append(scipy.sparse.csr_matrix((varClause,(negclauselist*2, vlist + vneglist)),shape=(self.nvar[findex], self.maxvar*2),dtype=np.float))

        print("total number of samples: " + str(sum(self.nsamples)))


    def __len__(self):
        return len(self.data)

    def mpermute(self,ssm, labels, pbid):
        res = ssm
        new_labels = []
        if self.permute_vars:
            if self.neg_as_link:
                varPermuted = np.random.permutation(list(range(self.maxvar)))
            else:
                varPermuted = np.random.permutation(list(range(self.maxvar*2)))
            if not self.varvar:
                if self.neg_as_link:
                    vperMatrix = scipy.sparse.csr_matrix((self.varData,(self.varPermRows,varPermuted)),shape=(self.maxvar,self.maxvar),dtype=float)
                else:
                    vperMatrix = scipy.sparse.csr_matrix((self.varData,(self.varPermRows,varPermuted)),shape=(self.maxvar*2,self.maxvar*2),dtype=float)
                res =  ssm * vperMatrix
            else :
                vperMatrix = scipy.sparse.csr_matrix((self.varData[pbid],(self.varPermRows,varPermuted)),shape=(self.maxvar*2,self.maxvar*2),dtype=float)
                res = vperMatrix * ssm * vperMatrix

            var_pos = []
            if self.neg_as_link:
                new_labels = [ (-varPermuted[-l-1]+1)
                               if (l< 0)
                               else (varPermuted[l-1]+1)
                               for l in labels]
            else:
                var_pos = [varPermuted[conv_vindex(l,self.maxvar,False)] for l in labels]
                new_labels = [(vp+1) if vp < self.maxvar else (-(vp-self.maxvar+1)) for vp in var_pos]
        else:
            new_labels = labels



        if self.permute_clauses:
            clausesPermuted = np.random.permutation(list(range(self.nclause[pbid])))
            clausePerMatrix = scipy.sparse.csr_matrix((self.varClause[pbid],(self.clausesPerRows[pbid],clausesPermuted)),shape=(self.nclause[pbid],self.nclause[pbid]),dtype=float)
            res = clausePerMatrix *res


        return res, new_labels

    def addNegClauses(self,ssm,pbid):
        if self.neg_as_link:
            return ssm
        if self.varvar:
            return ssm+self.negClauseMatrix[pbid]
        else:
            return scipy.sparse.vstack([ssm, self.negClauseMatrix[pbid]])


    def __getitem__(self, idx):
        if idx in self.cache:
            ssm, labels = self.mpermute(self.cache[idx][0], self.cache[idx][1], self.cache[idx][2])
            return ssm, labels, self.cache[idx][3]
        ssm = None
        for graphfile in self.data[idx]['f']:
            newssm = scipy.sparse.load_npz(self.path_prefix+graphfile).tocsr()
            shape = newssm.shape
            nvars = shape[1]
            if self.neg_as_link:
                newshape = (shape[0], self.maxvar)
                newssm.resize(newshape)
            else:
                posmatrix = newssm.tocsc()[:,:int(nvars/2)]
                negmatrix = newssm.tocsc()[:,int(nvars/2):]
                zeros = scipy.sparse.csc_matrix((shape[0],self.maxvar-int(nvars/2)),dtype=np.float)
                newssm = scipy.sparse.hstack([posmatrix,zeros,negmatrix,zeros]).tocsr()
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
            self.cache[idx] = (ssm, self.data[idx]['labels'],self.data[idx]['id'], nvars)
        if self.neg_clauses:
            ssm = self.addNegClauses(ssm,self.data[idx]['id'])
        ssm,labels =  self.mpermute(ssm, self.data[idx]['labels'], self.data[idx]['id'])
        return ssm, labels, nvars

    def getitem(self,idx):
        return self.__getitem__(idx)


    def getDataLoaders(self, batch_size, test_split,  maxclause, maxvar, varvar = True, graph_pool = False, num_workers=1):
        #dset = GraphDataset(filename, cachesize=cachesize)
        dataset_size = len(self.data)
        indices =list(range(dataset_size))
        split = int(np.floor(test_split*dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(self, batch_size= batch_size, num_workers=num_workers,pin_memory=False, collate_fn = lambda x : postproc(x, maxclause, maxvar, self.varvar, self.normalize, graph_pool, self.neg_as_link), sampler = train_sampler)

        test_loader = torch.utils.data.DataLoader(self, batch_size= batch_size, num_workers=num_workers,pin_memory=False, collate_fn = lambda x : postproc(x, maxclause, maxvar, self.varvar, graph_pool, self.neg_as_link), sampler = test_sampler)

        anl = self.totalNLabels/sum(self.nsamples)
        print("average number of labels: " + str(anl))
        print("average number of vars: " + str((sum(self.nvar)/len(self.nvar))))
        pos_weight = ((sum(self.nvar)/len(self.nvar))-anl)/anl
        neg_weight = 1.0
        if self.neg_as_link:
            weights = [neg_weight*3,(1-neg_weight)/2*3,(1-neg_weight)/2*3]
        else:
            weights = [neg_weight,pos_weight]
        return train_loader, test_loader, weights


def main():
    # start = time.process_time()
    # preprocess.preprocess('/data1/infantes/systerel/T102.2.1.cnf', './test', 10)
    # end = time.process_time()
    # print("preprocess time: " + str(end-start))
    start = time.process_time()
    tds = GraphDataset(['./debug/ex.graph'],  5, neg_clauses = True,permute_vars=True, self_supervised = False, cachesize=0)
    end = time.process_time()
    print("init time: " + str(end-start))
    start = time.process_time()
    ssm, labels, nvars = tds.getitem(0)
    print("shape: " + str(ssm.shape))
    print("ssm: \n" + str(ssm.todense()))
    print("labels: "+ str(labels))
    print("nvars: " + str(nvars))
    print(labels)
    end = time.process_time()
    print("get item time: " + str(end-start))
    print(ssm.shape)

    ssm, labels, nvars = tds.getitem(1)
    print("labels"+str(labels))
    print("ssm shame" + str(ssm.shape))
    print("ssm: \n" + str(ssm.todense()))
    print("nvars: " + str(nvars))
    end = time.process_time()
    print("get item time: " + str(end-start))

    # ssm, labels, nvars = tds.getitem(2)
    # print("labels"+str(labels))
    # print("ssm shame" + str(ssm.shape))
    # print("ssm: " + str(ssm.todense()))
    # print("nvars: " + str(nvars))
    # end = time.process_time()
    # print("get item time: " + str(end-start))


    # ssm, labels, nvars = tds.getitem(3)
    # print("labels"+str(labels))
    # print("ssm shame" + str(ssm.shape))
    # print("ssm: " + str(ssm.todense()))
    # print("nvars: " + str(nvars))
    # end = time.process_time()
    # print("get item time: " + str(end-start))


    # ssm, labels = tds.getitem(2)
    # print("labels")
    # print(labels)
    # end = time.process_time()
    # print("get item time: " + str(end-start))
    # print(ssm.shape)


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
