import similaripy as sim
from scipy import sparse
from datetime import datetime
from memory_profiler import profile

@profile
def run(sparse_matrix: sparse.csr_matrix, k):
    start_time = datetime.now()
    _ = sim.cosine(sparse_matrix, k=k, verbose=False)
    end_time = datetime.now()
    search_time = (end_time - start_time).total_seconds() * 1000
    return search_time

if __name__ == "__main__":
    k = 50

    for num_rows in [100, 1000, 10000]:
        sparse_matrix = sparse.random(
            num_rows, 10000, density=0.1, format='csr', random_state=42)
        search_time = run(sparse_matrix, k)
        print(search_time)
    
    movielens = sparse.load_npz('../data/movielens.npz')
    search_time = run(movielens, k)
    print(search_time)

    lastfm = sparse.load_npz('../data/lastfm.npz')
    search_time = run(lastfm, k)
    print(search_time)