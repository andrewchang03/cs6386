import numpy as np
from scipy.sparse import load_npz

U = load_npz("data/movielens/movielens.npz")

data = np.array(U.data, dtype='float32')
indices = np.array(U.indices, dtype='int32')
indptr = np.array(U.indptr, dtype='int32')

data.tofile('data/movielens/movielens-data')
indices.tofile('data/movielens/movielens-colinds')
indptr.tofile('data/movielens/movielens-rowptrs')