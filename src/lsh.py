import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
        cosines = cosine_similarity(query, self.data)
        top_k = np.argsort(cosines)
        top_k = top_k[:,::-1][:,:k]
        return top_k
    
    def unlearn(self, r, c=None, val=None):
        pass

class MultiProbeLSH(LSH):
    def __init__(self, nbits, dim, flips):
        super().__init__(nbits, dim)
        self.buckets = defaultdict(set)
        self.flips = flips
    
    def index(self, data):
        super().index(data)
        for i in range(len(self.hashed)):
            self.buckets[tuple(self.hashed[i])].add(i)

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
        query_code = self.hash(query)
        dists = [hamming(query_code, code) for code in self.hashed]
        sorted_indices = np.argsort(dists)
        top_k = sorted_indices[:k]
        return top_k, len(self.data)

def compute_metrics(exact_idxs, lsh_idxs):
    intersection = len(set(exact_idxs) & set(lsh_idxs))
    precision = intersection / len(lsh_idxs) if len(lsh_idxs) > 0 else 0
    recall = intersection / len(exact_idxs) if len(exact_idxs) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return np.array([precision, recall, f1_score])

def experiment(data, exact, lsh, search_iters=10):
    timer = util.Timer()
    lsh.index(data)
    index_time = timer.measure()

    avg_stats = np.zeros(3)
    avg_candidates = avg_search_time = 0

    for iter in range(search_iters):
        print('Iteration', iter)
        i = int(np.random.random() * data.shape[0])
        query = data[i]

        timer.start()
        approx, num_candidates = lsh.search(query, k=50)
        avg_search_time += timer.measure()
        avg_candidates += num_candidates

        stats = compute_metrics(exact[i], approx)
        avg_stats += stats

    avg_stats /= search_iters
    avg_candidates /= search_iters
    avg_search_time /= search_iters

    return avg_stats, avg_candidates, index_time, avg_search_time

def multi_probe_experiment(dataset, nbits, flips):
    data = util.load_npz('../data/' + dataset + '.npz')
    lsh = MultiProbeLSH(nbits=nbits, dim=data.shape[1], flips=flips)
    exact = np.load('../results/' + dataset + '_top50.npy')
    return experiment(data, exact, lsh)

def no_bucket_experiment(dataset, nbits):
    data = util.load_npz('../data/' + dataset + '.npz')
    lsh = NoBucketLSH(nbits=nbits, dim=data.shape[1])
    exact = np.load('../results/' + dataset + '_top50.npy')
    return experiment(data, exact, lsh)

def multi_probe_varying_nbits(dataset):
    nbits = [8, 12, 16, 20, 24, 28, 32, 36, 48]
    precisions = []
    recalls = []
    fs = []
    num_candidates = []
    index_times = []
    search_times = []

    for nbit in nbits:
        print('Number of Bits', nbit)
        avg_stats, avg_candidates, index_time, avg_search_time = multi_probe_experiment(dataset, nbit, 4)
        precisions.append(avg_stats[0])
        recalls.append(avg_stats[1])
        fs.append(avg_stats[2])
        num_candidates.append(avg_candidates)
        index_times.append(index_time)
        search_times.append(avg_search_time)

    plt.plot(nbits, precisions, label='Precision')
    plt.plot(nbits, recalls, label='Recall')
    plt.plot(nbits, fs, label='F-score')
    plt.xlabel('Number of LSH Hash Bits')
    plt.ylabel('Value')
    plt.title('Precision, Recall, F-score vs Number of LSH Hash Bits on ' + dataset)
    plt.legend()
    plt.savefig('../results/multi_probe_nbits_' + dataset + '.png')

    plt.clf()

    plt.plot(nbits, index_times, label='Indexing Time')
    plt.plot(nbits, search_times, label='Search TIme')
    plt.xlabel('Number of LSH Hash Bits')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime vs Number of LSH Hash Bits on ' + dataset)
    plt.legend()
    plt.savefig('../results/multi_probe_nbits_runtimes_' + dataset + '.png')

    print('Index Runtimes:', index_times)
    print('Search Runtimes:', search_times)

def multi_probe_varying_probes(dataset):
    probes = [1, 2, 4, 6, 8, 10, 12]
    precisions = []
    recalls = []
    fs = []
    num_candidates = []
    index_times = []
    search_times = []

    for probe in probes:
        print('Number of Probes', probe)
        avg_stats, avg_candidates, index_time, avg_search_time = multi_probe_experiment(dataset, 16, probe)
        precisions.append(avg_stats[0])
        recalls.append(avg_stats[1])
        fs.append(avg_stats[2])
        num_candidates.append(avg_candidates)
        index_times.append(index_time)
        search_times.append(avg_search_time)

    plt.plot(probes, precisions, label='Precision')
    plt.plot(probes, recalls, label='Recall')
    plt.plot(probes, fs, label='F-score')
    plt.xlabel('Number of Probes')
    plt.ylabel('Value')
    plt.title('Precision, Recall, F-score vs Number of Probes on ' + dataset)
    plt.legend()
    plt.savefig('../results/multi_probe_probes_' + dataset + '.png')

    plt.clf()

    plt.plot(probes, index_times, label='Indexing Time')
    plt.plot(probes, search_times, label='Search TIme')
    plt.xlabel('Number of Probes')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime vs Number of Probes on ' + dataset)
    plt.legend()
    plt.savefig('../results/multi_probe_probes_runtimes_' + dataset + '.png')

    print('Index Runtimes:', index_times)
    print('Search Runtimes:', search_times)

def no_bucket_varying_nbits(dataset):
    nbits = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    precisions = []
    index_times = []
    search_times = []

    for nbit in nbits:
        print('Number of Bits', nbit)
        avg_stats, avg_candidates, index_time, avg_search_time = no_bucket_experiment(dataset, nbit)
        precisions.append(avg_stats[0])
        index_times.append(index_time)
        search_times.append(avg_search_time)

    plt.plot(nbits, precisions)
    plt.xlabel('Number of LSH Hash Bits')
    plt.ylabel('Value')
    plt.title('Precision vs Number of LSH Hash Bits on ' + dataset)
    plt.legend()
    plt.savefig('../results/no_bucket_nbits_' + dataset + '.png')

    plt.clf()

    plt.plot(nbits, index_times, label='Indexing Time')
    plt.plot(nbits, search_times, label='Search TIme')
    plt.xlabel('Number of LSH Hash Bits')
    plt.ylabel('Runtime (ms)')
    plt.title('Runtime vs Number of LSH Hash Bits on ' + dataset)
    plt.legend()
    plt.savefig('../results/no_bucket_nbits_runtimes_' + dataset + '.png')

    print('Index Runtimes:', index_times)
    print('Search Runtimes:', search_times)

if __name__ == "__main__":
    np.random.seed(42)    

    # avg_stats, avg_candidates, index_time, avg_search_time = multi_probe_experiment('lastfm', 90, 4)

    # print(f"Precision: {avg_stats[0]:.4f}, Recall: {avg_stats[1]:.4f}, F1: {avg_stats[2]:.4f}")
    # print('Average Number of Candidates', avg_candidates)
    # print('Index Time', index_time)
    # print('Average Search Time', avg_search_time)

    # multi_probe_varying_nbits('movie')
    # multi_probe_varying_nbits('lastfm')
    # multi_probe_varying_probes('movie')
    # multi_probe_varying_probes('lastfm')
    # no_bucket_varying_nbits('movie')
    # no_bucket_varying_nbits('lastfm')