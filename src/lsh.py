import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity

import util

class LSH:
    def __init__(self, nbits, dim):
        self.nbits = nbits
        self.dim = dim
        self.hyperplanes = np.random.randn(nbits, dim)
        self.data = None
        self.hashed = None

    def hash(self, data):
        projection = data @ self.hyperplanes.T
        return (projection > 0).astype(np.uint8)
    
    def index(self, data):
        self.data = data
        self.hashed = self.hash(data)

    def search(self, query, k=50):
        pass

    def exact_search(self, query, k=50):
        cosines = cosine_similarity(query, data)
        top_k = np.argsort(cosines)
        top_k = top_k[:,::-1]
        top_k = top_k[:,:k]
        return util.sklearn_flat(query, self.data, k=k)

class MultiProbeLSH(LSH):
    def __init__(self, nbits, dim, flips):
        super().__init__(nbits, dim)
        self.buckets = defaultdict(list)
        self.flips = flips
    
    def index(self, data):
        super().index(data)
        for i in range(len(self.hashed)):
            self.buckets[tuple(self.hashed[i])].append(i)

    def multi_probe(self, base_code,):
        probes = set()
        bits = np.array(base_code)
        indices = list(range(self.nbits))

        for k in range(1, self.flips + 1):
            for flip_indices in combinations(indices, k):
                flipped = bits.copy()
                flipped[list(flip_indices)] = 1 - flipped[list(flip_indices)]
                probes.add(tuple(flipped))

        return list(probes)

    def search(self, query, k=50):
        code = tuple(self.hash(query))
        probe_buckets = [code] + self.multi_probe(code)

        candidate_idxs = []
        for probe in probe_buckets:
            candidate_idxs.extend(self.buckets[probe])

        if not candidate_idxs:
            return None

        candidates = self.data[candidate_idxs]
        
        cosines = cosine_similarity(query.reshape(1, -1), candidates).flatten()
        sorted_indices = np.argsort(cosines)[::-1][:k]
        top_k = np.array(candidate_idxs)[sorted_indices]
        return top_k, len(candidate_idxs)
    
class NoBucketLSH(LSH):
    def __init__(self, nbits, dim):
        super().__init__(nbits, dim)

    def search(self, query, k=50):
        query_code = self.hash(query[np.newaxis, :])[0]
        dists = [hamming(query_code, code) for code in self.hashed]
        top_k = np.argsort(dists)[:k]
        return top_k, len(self.data)

def compute_metrics(exact_idxs, lsh_idxs):
    intersection = len(set(exact_idxs) & set(lsh_idxs))
    precision = intersection / len(lsh_idxs) if len(lsh_idxs) > 0 else 0
    recall = intersection / len(exact_idxs) if len(exact_idxs) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return np.array([precision, recall, f1_score])

if __name__ == "__main__":
    np.random.seed(42)

    dataset = 'lastfm'
    data = util.load_npz('../data/' + dataset + '.npz')
    n, dim = data.shape
    query = data[178]
    exact = np.load('../results/' + dataset + '_top50.npy')

    lsh = MultiProbeLSH(nbits=12, dim=dim, flips=4)
    # lsh = NoBucketLSH(nbits=256, dim=dim)
    lsh.index(data)

    avg_stats = np.zeros(3)
    avg_candidates = 0
    iters = 50

    for iter in range(iters):
        print(iter)
        i = int(np.random.random() * n)
        query = data[i]

        approx, num_candidates = lsh.search(query, k=50)

        stats = compute_metrics(exact[i], approx)
        avg_stats += stats
        avg_candidates += num_candidates

    avg_stats /= iters
    avg_candidates /= iters

    print(f"Precision: {avg_stats[0]:.4f}, Recall: {avg_stats[1]:.4f}, F1: {avg_stats[2]:.4f}")
    print(avg_candidates)