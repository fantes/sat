import argparse
import time
import os

import scipy.sparse
import numpy as np

def conv_vindex(v, maxvar):
    if v<0:
        return -v+maxvar
    return v


def preprocess(cnffile,target_dir,nfiles,maxvar=5000000):
    basename = os.path.basename(cnffile)
    noext = os.path.splitext(basename)[0]
    matrixfiles = []

    with open(cnffile, 'r') as f:

        p = f.readline().strip().split()
        while p[0] == 'c':
            p = f.readline().strip().split()

        nvars = int(p[2])
        nclauses = int(p[3])
        print('number of vars: ' + str(nvars))
        print('number of clauses: ' + str(nclauses))

        start = time.process_time()

        clauseidx = 0
        numClausesPerFile = nclauses//nfiles
        for i in range(nfiles-1):
            totalnvc = 0
            indices0 = []
            indices1 = []
            values=[]
            for nc in range(clauseidx, numClausesPerFile):
                c = f.readline().strip().split()
                totalnvc += len(c)-1
                indices0.extend([nc]*(len(c)-1))
                indices1.extend([conv_vindex(int(v),maxvar) for v in c[:-1]])

            values = [True] * (totalnvc)
            smatrix = scipy.sparse.coo_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.bool,shape=(numClausesPerFile,maxvar*2+1))
            mf = target_dir+"/"+noext+"_"+str(i)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)
            clauseidx = clauseidx+numClausesPerFile


        if clauseidx < nclauses:
            totalnvc = 0
            indices0 = []
            indices1 = []
            values=[]
            for nc in range(0,nclauses-clauseidx):
                c = f.readline().strip().split()
                totalnvc += len(c)-1
                indices0.extend([nc]*(len(c)-1))
                indices1.extend([conv_vindex(int(v),maxvar) for v in c[:-1]])

            values = [True] * (totalnvc)
            smatrix = scipy.sparse.coo_matrix((np.asarray(values),(np.asarray(indices0),np.asarray(indices1))),dtype=np.bool,shape=(nclauses-clauseidx,maxvar*2+1))
            mf = target_dir+"/"+noext+"_"+str(nfiles-1)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)

        datafile = open(target_dir+"/"+noext+".graph","w")
        datafile.write("1 "+str(maxvar*2+1)+" " +str(nclauses)+ " " + str(nvars) + "\n")
        datafile.write("2")
        for fn in matrixfiles:
            datafile.write(" "+fn)
        datafile.close()





def main():
    parser = argparse.ArgumentParser(description='cnf reader')
    parser.add_argument('--filename', type=str, default="", help='file to parse')
    parser.add_argument('--nfiles', type=int, default=10, help='number of graphs')
    parser.add_argument('--target', type=str, default=".", help='target dir')
    args = parser.parse_args()

    preprocess(args.filename, args.target, args.nfiles)

if __name__ == '__main__' :
    main()
