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
    
    def unlearn(self):
        pass

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

def experiment(data, exact, lsh):
    n, dim = data.shape
    timer = util.Timer()
    lsh.index(data)
    index_time = timer.measure()

    avg_stats = np.zeros(3)
    avg_candidates = avg_search_time = 0
    iters = 50

    for iter in range(iters):
        print('Iteration', iter)
        i = int(np.random.random() * n)
        query = data[i]

        timer.start()
        approx, num_candidates = lsh.search(query, k=50)
        avg_search_time += timer.measure()

        stats = compute_metrics(exact[i], approx)
        avg_stats += stats
        avg_candidates += num_candidates

    avg_stats /= iters
    avg_candidates /= iters
    avg_search_time /= iters

    return avg_stats, avg_candidates, index_time, avg_search_time

def multi_probe_experiment(dataset, nbits, flips):
    data = util.load_npz('../data/' + dataset + '.npz')
    exact = np.load('../results/' + dataset + '_top50.npy')

    lsh = MultiProbeLSH(nbits=nbits, dim=data.shape[1], flips=flips)
    return experiment(data, exact, lsh)

def no_bucket_experiment(dataset, nbits):
    data = util.load_npz('../data/' + dataset + '.npz')
    exact = np.load('../results/' + dataset + '_top50.npy')

    lsh = NoBucketLSH(nbits=nbits, dim=data.shape[1])
    return experiment(data, exact, lsh)

if __name__ == "__main__":
    np.random.seed(42)

    # avg_stats, avg_candidates, index_time, avg_search_time = multi_probe_experiment('lastfm', 12, 4)
    avg_stats, avg_candidates, index_time, avg_search_time = no_bucket_experiment('lastfm', 256)

    print(f"Precision: {avg_stats[0]:.4f}, Recall: {avg_stats[1]:.4f}, F1: {avg_stats[2]:.4f}")
    print('Average Number of Candidates', avg_candidates)
    print('Index Time', index_time)
    print('Average Search Time', avg_search_time)