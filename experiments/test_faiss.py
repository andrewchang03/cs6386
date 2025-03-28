import faiss
from scipy import sparse
from datetime import datetime
from memory_profiler import profile

@profile
def run(sparse_matrix: sparse.csr_matrix):
    dense_matrix = sparse_matrix.toarray().astype('float32')

    start_time = datetime.now()
    index = faiss.IndexFlatIP(dense_matrix.shape[1])  # cosine similarity
    index.add(dense_matrix)
    end_time = datetime.now()
    index_time = (end_time - start_time).total_seconds() * 1000
    # print('index time:', index_time)

    start_time = datetime.now()
    scores, indices = index.search(dense_matrix, k=50)  # pairwise similarities
    end_time = datetime.now()
    search_time = (end_time - start_time).total_seconds() * 1000
    # print('search time:', search_time)

    return index_time, search_time

if __name__ == "__main__":
    for num_rows in [100, 1000, 10000]:
        sparse_matrix = sparse.random(
            num_rows, 10000, density=0.1, format='csr', random_state=42)
        index_time, search_time = run(sparse_matrix)
        print(index_time, search_time)
    
    movielens = sparse.load_npz('../data/movielens.npz')
    index_time, search_time = run(movielens)
    print(index_time, search_time)

    lastfm = sparse.load_npz('../data/lastfm.npz')
    index_time, search_time = run(lastfm)
    print(index_time, search_time)