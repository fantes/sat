import argparse
import time
import os

import scipy.sparse
import numpy as np

def conv_vindex(v, maxvar):
    if v<0:
        return -v+maxvar
    return v


def create_dummy(cnffile,target_dir,nfiles,maxvar=5000000, maxclauses=20000000):
    basename = os.path.basename(cnffile)
    noext = os.path.splitext(basename)[0]
    matrixfiles = []

    with open(cnffile, 'r') as f:

        p = f.readline().strip().split()
        while p[0] == 'c':
            p = f.readline().strip().split()

        start = time.process_time()
        nvars = int(p[2])
        nclauses = int(p[3])
        print('number of vars: ' + str(nvars))
        print('number of nclauses: ' + str(nclauses))

        indices0 = []
        indices1 = []
        values=[]

        # vlist = [v for v in range(1,nvars+1)]
        # vneglist = [maxvar+v for v in range(1,nvars+1)]
        # negclauselist = [2*maxvar+v for v in range(1,nvars+1)]

        # indices0 = vlist + vneglist
        # indices1 = negclauselist * 2

        end = time.process_time()
        print("reading var time:" + str(end-start))
        start = time.process_time()

        totalnvc = 0

        for nc in range(nclauses):
            c = f.readline().strip().split()
            totalnvc += len(c)-1
            indices0.extend([conv_vindex(int(v),maxvar) for v in c[:-1]])
            indices1.extend([2*maxvar+nc]*(len(c)-1)) #clause index starts after pos and neg vars

        #values = [True] * (2*nvars + totalnvc)
        values = [True] * (totalnvc)

        end = time.process_time()
        print("reading clauses time: " + str(end-start))

        step = int(totalnvc / nfiles)

        for i in range(nfiles-1):
            smatrix = scipy.sparse.coo_matrix((np.asarray(values[i*step:(i+1)*step]),(np.asarray(indices0[i*step:(i+1)*step]),np.asarray(indices1[i*step:(i+1)*step]))),dtype=np.bool,shape=(maxvar+maxclauses, maxvar+maxclauses))
            mf = target_dir+"/"+noext+"_"+str(i)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)
        last = step * (nfiles-1)
        if last < totalnvc:
            smatrix = scipy.sparse.coo_matrix((np.asarray(values[last:]),(np.asarray(indices0[last:]),np.asarray(indices1[last:]))),dtype=np.bool,shape=(maxvar+maxclauses, maxvar+maxclauses))
            mf = target_dir+"/"+noext+"_"+str(nfiles-1)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)

        datafile = open(target_dir+"/"+noext+".graph","w")
        datafile.write("1 "+str(maxvar)+" " +str(maxclauses))
        for fn in matrixfiles:
            datafile.write(" "+fn)
        datafile.close()





def main():
    parser = argparse.ArgumentParser(description='cnf reader')
    parser.add_argument('--filename', type=str, default="", help='file to parse')
    parser.add_argument('--nfiles', type=int, default=10, help='number of graphs')
    parser.add_argument('--target', type=str, default=".", help='target dir')
    args = parser.parse_args()

    create_dummy(args.filename, args.target, args.nfiles)

if __name__ == '__main__' :
    main()
