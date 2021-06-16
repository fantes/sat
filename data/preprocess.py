import argparse
import time
import os

import scipy.sparse
import numpy as np
import pickle
import tqdm

def conv_vindex(v, nvar, neg_as_link):
    if neg_as_link:
        if v<0:
            return -v-1
        return v-1
    else:
        if v<0:
            return -v+nvar-1
        return v-1


def preprocess(varvar, neg_as_link, cnffile,target_dir,nfiles, save_arity):
    basename = os.path.basename(cnffile)
    noext = os.path.splitext(basename)[0]
    matrixfiles = []

    try:
        os.mkdir(target_dir)
    except:
        pass

    with open(cnffile, 'r') as f:

        p = f.readline().strip().split()
        while p[0] == 'c':
            p = f.readline().strip().split()

        nvars = int(p[2])
        nclauses = int(p[3])
        print('number of vars: ' + str(nvars))
        print('number of clauses: ' + str(nclauses))

        if save_arity:
            if neg_as_link:
                varArity = [0]*nvars
            else:
                varArity = [0]*nvars*2
            clauseArity = [0] * nclauses

        start = time.process_time()

        clauseidx = 0
        numClausesPerFile = nclauses//nfiles
        for i in tqdm.tqdm(range(nfiles-1)):
            totalnvc = 0
            indices0 = []
            indices1 = []
            values=[]
            for nc in range(0, numClausesPerFile):
                c = f.readline().strip().split()[:-1]
                if not varvar:
                    totalnvc += len(c)
                    indices0.extend([nc]*(len(c)))
                    varidx = [conv_vindex(int(v),nvars,neg_as_link) for v in c]
                    indices1.extend(varidx)
                    if neg_as_link:
                        for v in c:
                            if int(v)<0:
                                values.append(-1)
                            else:
                                values.append(1)
                    else:
                        values.extend([True]*len(c))

                    if save_arity:
                        clauseArity[clauseidx] = len(c)
                        for v in varidx:
                            varArity[v] += 1
                else:
                    totalnvc += int(len(c) * (len(c)-1)/2)
                    c = [conv_vindex(int(v),nvars,neg_as_link) for v in c]
                    for v1 in range(0, len(c)):
                        for v2 in range(v1+1, len(c)):
                            i1 = c[v1]
                            i2 = c[v2]
                            if i1 < i2:
                                indices0.append(i1)
                                indices1.append(i2)
                            else:
                                indices0.append(i2)
                                indices1.append(i1)
                            values.append(True)
                    if save_arity:
                        for v in c:
                            varArity[v] += len(c)-1
                clauseidx += 1

            if not varvar:
                if neg_as_link:
                    smatrix = scipy.sparse.csr_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.byte,shape=(numClausesPerFile,nvars))
                else:
                    smatrix = scipy.sparse.csr_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.bool,shape=(numClausesPerFile,nvars*2))
            else:
                smatrix = scipy.sparse.csr_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.bool,shape=(nvars*2,nvars*2))

            mf = target_dir+"/"+noext+"_"+str(i)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)
            #clauseidx = clauseidx+numClausesPerFile


        if clauseidx < nclauses:
            lastclauseidx = clauseidx
            totalnvc = 0
            indices0 = []
            indices1 = []
            values=[]
            for nc in range(0,nclauses-clauseidx):
                c = f.readline().strip().split()[:-1]
                if not varvar:
                    totalnvc += len(c)
                    indices0.extend([nc]*(len(c)))
                    varidx = [conv_vindex(int(v),nvars,neg_as_link) for v in c]
                    indices1.extend(varidx)
                    if neg_as_link:
                        for v in c:
                            if int(v)<0:
                                values.append(-1)
                            else:
                                values.append(1)
                    else:
                        values.extend([True]*len(c))
                    if save_arity:
                        clauseArity[clauseidx] = len(c)
                        for v in varidx:
                            varArity[v] += 1
                else:
                    totalnvc += int(len(c) * (len(c)-1)/2)
                    c = [conv_vindex(int(v),nvars,neg_as_link) for v in c]
                    for v1 in range(0, len(c)):
                        for v2 in range(v1+1, len(c)):
                            i1 = c[v1]
                            i2 = c[v2]
                            if i1 < i2:
                                indices0.append(i1)
                                indices1.append(i2)
                            else:
                                indices0.append(i2)
                                indices1.append(i1)
                            values.append(True)
                    if save_arity:
                        for v in c:
                            varArity[v] += len(c)-1
                clauseidx += 1

            if not varvar:
                if neg_as_link:
                    smatrix = scipy.sparse.csr_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.byte,shape=(nclauses-lastclauseidx,nvars))
                else:
                    smatrix = scipy.sparse.csr_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.bool,shape=(nclauses-lastclauseidx,nvars*2))
            else:
                smatrix = scipy.sparse.csr_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.bool,shape=(nvars*2,nvars*2))

            mf = target_dir+"/"+noext+"_"+str(nfiles-1)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)

        datafile = open(target_dir+"/"+noext+".graph","w")
        datafile.write("1 " +str(nclauses)+ " " + str(nvars) + "\n")
        datafile.write("2") # example label
        for fn in matrixfiles:
            datafile.write(" "+fn)
        datafile.close()

        if save_arity:
            varArityFile = open(target_dir+"/"+noext+".varArity","wb")
            pickle.dump(varArity, varArityFile)
            clauseArityFile = open(target_dir+"/"+noext+".clauseArity","wb")
            pickle.dump(clauseArity, clauseArityFile)





def main():
    parser = argparse.ArgumentParser(description='cnf reader')
    parser.add_argument('--filename', type=str, default="", help='file to parse')
    parser.add_argument('--nfiles', type=int, default=10, help='number of graphs')
    parser.add_argument('--target', type=str, default=".", help='target dir')
    parser.add_argument('--arity', action='store_true')
    parser.add_argument('--neg-as-link', action='store_true')
    parser.add_argument('--type', type=str, default="clause.var", help='graph type, in [clause.var|var.var]')
    args = parser.parse_args()

    if args.type == "clause.var":
        varvar=False
    elif args.type == "var.var":
        varvar=True
    else:
        print("unknown graph type")
        exit(-1)
    preprocess(varvar, args.neg_as_link, args.filename, args.target, args.nfiles, args.arity)

if __name__ == '__main__' :
    main()
