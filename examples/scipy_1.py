"""
Uses scipy to solve same system as ex_1.rs
"""
import numpy as np
import time
from scipy.sparse import *
from scipy.sparse.linalg import gmres

def ex_run(mat_f="./data/fidap001.txt", rhs_f="./data/fidap001_rhs1.txt", inner_itr=500, tol=1e-8):
    a_triplets = np.genfromtxt(mat_f)
    b_rhs = np.genfromtxt(rhs_f)

    # triplets into coo sparse matrix
    idx_i = np.asarray(a_triplets[:, 0]-1, dtype=int)
    idx_j = np.asarray(a_triplets[:, 1]-1, dtype=int)
    vals = np.asarray(a_triplets[:, 2], dtype=np.float64)

    a_sparse_coo = coo_matrix((vals, (idx_i, idx_j)))

    # convert to csc sparse matrix
    a_sparse = a_sparse_coo.tocsc(copy=True)

    x0 = np.zeros(len(b_rhs))
    ti = time.time()
    # note maxiter is nubmer of outer iters, restart is number on inners
    # scipy gmres is restarted_gmres by default
    res_x, info = gmres(a_sparse, b_rhs, x0=x0, tol=tol, restart=inner_itr, maxiter=1)
    print("solution:", res_x)
    print("Iterations: %d" % info)
    tf = time.time()
    print("Solve time: %0.4e (s)" % (tf-ti))

if __name__ == "__main__":
    ex_run(mat_f="./data/fidap001.txt", rhs_f="./data/fidap001_rhs1.txt", inner_itr=500, tol=1e-12)
    ex_run(mat_f="./data/e40r0100.txt", rhs_f="./data/e40r0100_rhs1.txt", inner_itr=5000, tol=1e-6)
