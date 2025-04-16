import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from bitarray import bitarray
import faiss
from datetime import datetime

def query_exact_top_k(q: sparse.csr_matrix, data: sparse.csr_matrix, k: int) -> np.ndarray:
    """
    Retrieves indices of top [k] similar rows of [data] to [q] based on cosine similarity.
    Returns numpy array of pairs containing (index, similarity)
    """
    cosines = cosine_similarity(q, data)
    top_k = np.argsort(cosines)
    top_k = top_k[:,::-1]
    top_k = top_k[:,1:k+1]
    return top_k



class LSH():
    def __init__(self, data: sparse.csr_matrix, nbits: int, k: int, repeat: int):
        self.data = data
        self.hyperplanes = np.random.randn(nbits, data.shape[1])
        self.buckets = defaultdict(list)

    def hamming_distance(self, tup1: tuple, tup2: tuple):
        pass

    def index(self):
        dot_products = data @ self.hyperplanes.T
        sides = dot_products > 0
        hashes = list(map(lambda side: tuple(bitarray(side)), sides.tolist()))
        for i in range(len(hashes)):
            self.buckets[hashes[i]].append(i)
        print(self.buckets)
        return hashes

# def lsh(data: sparse.csr_matrix, nbits: int, k: int, repeat: int):
#     candidates = defaultdict(set)

#     for _ in range(repeat):
#         hyperplanes = np.random.randn(nbits, data.shape[1])

#         dot_products = data @ hyperplanes.T
#         sides = dot_products > 0

#         hashes = list(map(lambda side: tuple(bitarray(side)), sides.tolist()))

#         buckets = defaultdict(list)
#         for i in range(len(hashes)):
#             buckets[hashes[i]].append(i)

#         # top_k = np.zeros((data.shape[0], k))

#         for hash in buckets:
#             for i in range(len(buckets[hash])):
#                 for j in range(i + 1, len(buckets[hash])):
#                     candidates[buckets[hash][i]].add(buckets[hash][j])
#                     candidates[buckets[hash][j]].add(buckets[hash][i])

#     return candidates

def test_faiss(sparse_matrix, nbits, k):
    dense_matrix = sparse_matrix.toarray().astype('float32')

    start_time = datetime.now()
    index = faiss.IndexFlatIP(dense_matrix.shape[1])
    # index.add(dense_matrix)
    # end_time = datetime.now()
    # index_time = (end_time - start_time).total_seconds() * 1000
    # # print('index time:', index_time)

    # start_time = datetime.now()
    # scores, indices = index.search(dense_matrix, k=k)
    # end_time = datetime.now()
    # search_time = (end_time - start_time).total_seconds() * 1000
    # # print('search time:', search_time)

    return scores, indices

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    n = 100
    d = 100
    k = 5
    nbits = 8

    data = sparse.random(n, d, density=0.1, format='csr', random_state=seed)

    seeds = np.random.choice(np.arange(1, 1001), size=1, replace=False)
    for seed in seeds:
        q = sparse.random(1, d, density=0.1, format='csr', random_state=seed)
        exact = query_exact_top_k(q, data, k)
        print(exact)

        lsh = LSH(data, nbits, k, 10)
        hash = lsh.index()
        # print(hash)

        # candidates = lsh(data, nbits, k, 10)
        # print(candidates)
        # for candidate in candidates:
        #     lst = candidates[candidate]
        #     print(lst)

    # for hash in buckets:
        #     subsamples = data[buckets[hash], :]
        #     cosines = cosine_similarity(subsamples, data)
        #     # print(cosines.shape)
        #     temp = np.argsort(cosines)
        #     temp = temp[:,::-1]
        #     temp = temp[:,:len(cosines) - k - 1:-1]
        #     print(temp)
        #     # print(hash, x.shape)

    # scores, indices = test_faiss(data, nbits, k)
    # print(index)
    # print(indices)
    # hyperplanes = random_hyperplanes(n_bits, data.shape[1])
    # print(hyperplanes)
    # buckets = generate_hash_buckets(data, hyperplanes, power_of_twos)
    # print(buckets)
    # hash = query_hash(q, hyperplanes, power_of_twos).item()
    # print(buckets[hash])