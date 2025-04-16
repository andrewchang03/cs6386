import faiss
from scipy import sparse
from datetime import datetime

def get_stats(indices1, indices2):
    pass

def search_exact(sparse_matrix, k):
    dense_matrix = sparse_matrix.toarray().astype('float32')
    n, d = dense_matrix.shape

    start_time = datetime.now()
    index = faiss.IndexFlatIP(d)
    index.add(dense_matrix)
    tindex = (datetime.now() - start_time).total_seconds() * 1000

    start_time = datetime.now()
    scores, indices = index.search(dense_matrix, k=k)
    tsearch = (datetime.now() - start_time).total_seconds() * 1000

    return scores, indices, tindex, tsearch

def search_lsh(sparse_matrix, k):
    dense_matrix = sparse_matrix.toarray().astype('float32')
    n, d = dense_matrix.shape

    start_time = datetime.now()
    index = faiss.IndexLSH(d, 1000)
    index.add(dense_matrix)
    tindex = (datetime.now() - start_time).total_seconds() * 1000

    start_time = datetime.now()
    scores, indices = index.search(dense_matrix, k=k)
    tsearch = (datetime.now() - start_time).total_seconds() * 1000

    return scores, indices, tindex, tsearch

if __name__ == "__main__":
    # data = sparse.load_npz('../data/lastfm.npz')
    data = sparse.random(1000, 100, density=1, format='csr', random_state=42)
    k = 50
    
    scores_exact, indices_exact, tindex_exact, tsearch_exact = search_exact(data, k)
    # print(indices_exact)

    scores, indices, index_time, search_time = search_lsh(data, k)
    # print(indices)
    print(scores)

    total = 0

    for i in range(data.shape[0]):
        exact = indices_exact[i]
        approx = indices[i]
        counter = 0
        for idx in approx:
            if idx in exact:
                counter += 1
        # print(i, counter / 50)
        total += (counter / k)

    print(total / data.shape[0])

    # print(scores)
    # print(indices)
    # print(index_time, search_time)