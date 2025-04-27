import faiss
import numpy as np
from collections import defaultdict
from scipy import sparse
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
import os

import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class LSH:
    def __init__(self, nbits, dim):
        self.nbits = nbits
        self.dim = dim
        self.hyperplanes = np.random.randn(nbits, dim)
        self.data = None
        self.hashed = None
        self.removed_rows = set()

    def hash(self, data):
        projection = data @ self.hyperplanes.T
        return (projection > 0).astype(np.uint8)
    
    def index(self, data):
        self.data = data
        self.hashed = self.hash(data)

    def search(self, query, k=50):
        """
        Args:
            query: query vector of shape (d,)
            k: how many top k results we want
        """
        query_hash = self.hash(query)
        dists = []
        for i in range(len(self.hashed)):
            if i in self.removed_rows:
                dists.append(np.inf)
            else:
                dists.append(hamming(query_hash, self.hashed[i]))
        sorted_indices = np.argsort(dists)
        top_k = sorted_indices[:k]
        return top_k
    
    def unlearn(self, r, c=None, val=None):
        if not c:
            self.removed_rows.add(r)
        else:
            self.data[r][c] = val
            self.hashed[r] = self.hash(self.data[r])

    def exact_search(self, query, k=50):
        cosines = cosine_similarity(query, self.data)
        top_k = np.argsort(cosines)
        top_k = top_k[:,::-1][:,:k]
        return top_k

    def stats(self, exact_idxs, lsh_idxs):
        tp = len(set(exact_idxs) & set(lsh_idxs))  # true positives
        precision = tp / len(lsh_idxs) if len(lsh_idxs) > 0 else 0
        recall = tp / len(exact_idxs) if len(exact_idxs) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        return np.array([precision, recall, f1])

def experiment(dataset, nbits, search_iters=10):
    data = util.load_npz('../data/' + dataset + '.npz')
    lsh = LSH(nbits=nbits, dim=data.shape[1])
    exact = np.load('../results/' + dataset + '_top50.npy')

    timer = util.Timer()
    lsh.index(data)
    index_time = timer.measure()

    avg_stats = np.zeros(3)
    avg_search_time = 0

    for iter in range(search_iters):
        print('Iteration', iter)
        # i = int(np.random.random() * data.shape[0])
        i = 100
        query = data[i]
        j = 41
        # for j in range(len(query)):
        #     if query[j] != 0:
        #         print(j)
        #         break

        timer.start()
        approx = lsh.search(query, k=50)
        avg_search_time += timer.measure()

        timer.start()
        lsh.unlearn(i)
        # lsh.unlearn(i, j, 0)
        unlearn_time = timer.measure()

        stats = lsh.stats(exact[i], approx)
        avg_stats += stats

    avg_stats /= search_iters
    avg_search_time /= search_iters

    return avg_stats, index_time, avg_search_time

def no_bucket_varying_nbits(dataset):
    nbits = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    precisions = []
    index_times = []
    search_times = []

    for nbit in nbits:
        print('Number of Bits', nbit)
        avg_stats, index_time, avg_search_time = experiment(dataset, nbit)
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

    # data = sparse.load_npz('../data/movie.npz')
    # print(data)

    avg_stats, index_time, avg_search_time = experiment('lastfm', 512)

    print(f"Precision: {avg_stats[0]:.4f}, Recall: {avg_stats[1]:.4f}, F1: {avg_stats[2]:.4f}")
    print('Index Time', index_time)
    print('Average Search Time', avg_search_time)