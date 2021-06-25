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

def get_smatrix(varvar,neg_as_link,nclauses,nvars,v,i0,i1):
    if not varvar:
        if neg_as_link:
            smatrix = scipy.sparse.csr_matrix((np.asarray(v), (np.asarray(i0), np.asarray(i1))),
                                              dtype=np.byte,shape=(nclauses,nvars))
        else:
            smatrix = scipy.sparse.csr_matrix((np.asarray(v),(np.asarray(i0),np.asarray(i1))),
                                              dtype=np.bool,shape=(nclauses,nvars*2))
    else:
        smatrix = scipy.sparse.csr_matrix((np.asarray(v),(np.asarray(i0),np.asarray(i1))),
                                          dtype=np.bool,shape=(nvars*2,nvars*2))
    return smatrix


def get_indices_values_from_text_clause(nvars, neg_as_link, textclause,varvar,offsetnc):
    clause = [conv_vindex(int(v),nvars,neg_as_link) for v in textclause]
    return get_indices_values_from_clause(clause,varvar,neg_as_link, offsetnc)


def get_indices_values_from_clause(clause,varvar,neg_as_link, offsetnc):
    indices0=[]
    indices1=[]
    values=[]
    if not varvar:
        indices0.extend([offsetnc]*(len(clause)))
        indices1.extend(clause)
        if neg_as_link:
            for v in clause:
                if int(v)<0:
                    values.append(-1)
                else:
                    values.append(1)
        else:
            values.extend([True]*len(clause))

    else:
        for v1 in range(0, len(clause)):
            for v2 in range(v1+1, len(clause)):
                i1 = clause[v1]
                i2 = clause[v2]
                if i1 < i2:
                    indices0.append(i1)
                    indices1.append(i2)
                else:
                    indices0.append(i2)
                    indices1.append(i1)
                values.append(True)
    return indices0, indices1, values


def process_cnffile(cnffile,target_dir,nfiles,varvar,neg_as_link, save_arity=False):
    #parse cnffile
    varArity = None
    clauseArity = None
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
            indices0 = []
            indices1 = []
            values=[]

            for nc in range(0, numClausesPerFile):
                c = f.readline().strip().split()[:-1]
                newi0, newi1, newv = get_indices_values_from_text_clause(nvars,neg_as_link,
                                                                         c, varvar,nc)
                indices0.extend(newi0)
                indices1.extend(newi1)
                values.extend(newv)
                if save_arity:
                    varidx = [conv_vindex(int(v),nvars,neg_as_link) for v in c]
                    if not varvar:
                        clauseArity[clauseidx] = len(c)
                        for v in varidx:
                            varArity[v] += 1
                    else:
                        for v in varidx:
                            varArity[v] += len(c)-1
                clauseidx += 1

            smatrix = get_smatrix(varvar,neg_as_link, numClausesPerFile,
                                  nvars,values,indices0, indices1)

            mf = target_dir+"/"+noext+"_"+str(i)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)
            #clauseidx = clauseidx+numClausesPerFile

        #remaining clauses in case of multiple files
        if clauseidx < nclauses:
            lastclauseidx = clauseidx
            indices0 = []
            indices1 = []
            values=[]
            for nc in range(0,nclauses-clauseidx):
                c = f.readline().strip().split()[:-1]
                newi0, newi1, newv = get_indices_values_from_text_clause(nvars, neg_as_link,
                                                                         c, varvar,nc)
                indices0.extend(newi0)
                indices1.extend(newi1)
                values.extend(newv)
                if save_arity:
                    varidx = [conv_vindex(int(v),nvars,neg_as_link) for v in c]
                    if not varvar:
                        clauseArity[clauseidx] = len(c)
                        for v in varidx:
                            varArity[v] += 1
                    else:
                        for v in varidx:
                            varArity[v] += len(c)-1
                clauseidx += 1

            smatrix = get_smatrix(varvar,neg_as_link, nclauses-lastclauseidx, nvars,
                                  values, indices0, indices1)

            mf = target_dir+ "/" + noext+"_"+str(nfiles-1)+".npz"
            matrixfiles.append(mf)
            scipy.sparse.save_npz(mf, smatrix)

        return nclauses, nvars, matrixfiles, varArity, clauseArity

def process_arup(arupfile, target_dir, nvars,varvar,neg_as_link):
    # should return a list of delta_graph,[labels of forthcoming conflict]
    # delta_graph is delta to base cnf (can be empty)
    added_data = []
    current_matrix = None
    basename = os.path.basename(arupfile)
    noext = os.path.splitext(basename)[0]

    arupcount = 0
    with open(arupfile, 'r') as arup:
        lines = arup.readlines()
        for line in lines:
            l = line.strip().split()
            if l[0] == "c":
                continue
            if l[0] == "d":
                print("found delete:" + str(l))
                continue
            #new clause added:
            strvar = l[:-1]
            #record current conflict as labels
            labels = [int(v) for v in strvar]
            if current_matrix is None:
                added_data.append(["",labels])
            else:
                mf = target_dir+ "/" + noext+"_arup_"+str(arupcount)+".npz"
                scipy.sparse.save_npz(mf, current_matrix)
                added_data.append([mf,labels])
                arupcount += 1

            #append new clause to current matrix for future graphs/labels
            i0, i1, v = get_indices_values_from_text_clause(nvars, neg_as_link, strvar, varvar,0)
            smatrix = get_smatrix(varvar,neg_as_link, 1, nvars,v,i0, i1)
            if current_matrix is None:
                current_matrix = smatrix
            else:
                if varvar:
                    current_matrix += smatrix
                else:
                    current_matrix = scipy.sparse.vstack([current_matrix,smatrix])
    return added_data


def preprocess(varvar, neg_as_link, cnffile,arupfile, target_dir,nfiles, save_arity):

    try:
        os.mkdir(target_dir)
    except:
        pass

    nclauses, nvars , matrix_files,varArity, clauseArity = process_cnffile(cnffile,target_dir, nfiles,
                                                                           varvar,neg_as_link,
                                                                           save_arity)

    #read arup // TODO arity is not updated ATM
    data = process_arup(arupfile,target_dir, nvars,varvar,neg_as_link)

    #now dump real data
    basename = os.path.basename(cnffile)
    noext = os.path.splitext(basename)[0]
    datafile = open(target_dir+"/"+noext+".graph","w")
    datafile.write(str(len(data)) +" " +str(nclauses)+ " " + str(nvars) + "\n")
    for d  in data:
        datafile.write(str(len(d[1])))
        for l in d[1]:
            datafile.write(" " + str(l))
        for fn in matrix_files:
            datafile.write(" "+fn)
        datafile.write(" " + d[0])
        datafile.write("\n")

    datafile.close()

    if save_arity:
        varArityFile = open(target_dir+"/"+noext+".varArity","wb")
        pickle.dump(varArity, varArityFile)
        clauseArityFile = open(target_dir+"/"+noext+".clauseArity","wb")
        pickle.dump(clauseArity, clauseArityFile)


def main():
    parser = argparse.ArgumentParser(description='cnf reader')
    parser.add_argument('--cnf', type=str, default="", help='cnf file to parse')
    parser.add_argument('--arup', type=str, default="", help='arup file to parse')
    parser.add_argument('--nfiles', type=int, default=1, help='number of graphs')
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
    preprocess(varvar, args.neg_as_link, args.cnf, args.arup, args.target, args.nfiles, args.arity)

if __name__ == '__main__' :
    main()
