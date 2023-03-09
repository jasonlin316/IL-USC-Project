import numpy as np
from scipy import sparse
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return  loader['data'], loader['indices'], loader['indptr'],loader['shape']

def save_sparse_coo(filename, array):
    np.savez(filename, coo_data=array.data, row=array.row,
            col=array.col, shape=array.shape)

def load_sparse_coo(filename):
    loader = np.load(filename)
    return  loader['coo_data'], loader['row'], loader['col'],loader['shape']


#b=sparse.csr_matrix(a)
#save_sparse_csr('sparse', b)
#data, indicies, indptr, shape = load_sparse_csr('/home/jason/cluster-gcn/cluster_gcn/sparse_norm.npz')
data, indices, indptr, shape = load_sparse_csr('/home/jason/cluster-gcn/cluster_gcn/sparse_norm.npz')
print(shape)
new_csr = csr_matrix((data, indices, indptr), shape=shape)
b = new_csr.tocoo()
#b=sparse.coo_matrix(dense)
save_sparse_coo('coo_sparse',b)
coo_data, row, col, shape = load_sparse_coo('coo_sparse.npz')




print(coo_data)
print(row)
print(col)

coo_data.tofile('coo_data.bin')
row.tofile('row.bin')
col.tofile('col.bin')
