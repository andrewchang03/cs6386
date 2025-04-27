import numpy as np
from collections import defaultdict
from scipy import sparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class LSH:
    def __init__(self, num_bits, dim, device='cpu'):
        """
        Args:
            num_bits: number of random projections in LSH, or number of hash bits
            dim: number of dimensions of data
            device: devices to operate on for torch
        """
        self.num_bits = num_bits
        self.dim = dim
        self.device = torch.device(device)
        self.hyperplanes = torch.randn(num_bits, dim).to(device)
        self.data = None
        self.hashed = None
        self.removed_rows = set()
        self.started_time = datetime.now()

    def hash(self, data):
        """
        Args:
            data: torch.Tensor of shape (arbitrary length, dim)
        
        Returns:
            torch.Tensor of shape (arbitrary length, self.num_bits)
            with each row being a randomly projected hash
        """
        projection = data @ self.hyperplanes.T
        return projection > 0
    
    def index(self, data):
        """
        Hashes each row of [data]

        Args:
            data: torch.Tensor of shape (n, dim)
        """
        self.data = data
        self.hashed = self.hash(data)

    def hamming(self, query_hash, data_hash):
        """
        Args:
            query_hash: torch.Tensor of shape (self.num_bits,)
            data_hash: torch.Tensor of shape (n, self.num_bits)
        
        Returns:
            torch.Tensor of shape (n,) containing hamming distance between
            [query_hash] and each row of [data_hash]
        """
        return (query_hash != data_hash).sum(dim=1, dtype=torch.int32)

    def search(self, query, k=50):
        """
        Args:
            query: torch.Tensor query vector of shape (d,)
            k: how many top k results we want

        Returns:
            torch.Tensor of shape(k,) containing indices of [k] most
            similar rows of [self.hashed] in terms of hamming distance
        """
        query_hash = self.hash(query)
        dists = self.hamming(query_hash, self.hashed)
        dists[list(self.removed_rows)] = torch.iinfo(dists.dtype).max
        return torch.topk(dists, k, largest=False).indices
    
    def unlearn(self, r, c=None, val=None):
        """
        Args:
            r: row to unlearn
            c: if provided a column index, sets self.data[r][c] to [val] and rehashes. \
                otherwise, deletes row [r] from data
        """
        if not c:
            self.removed_rows.add(r)
        else:
            self.data[r][c] = val
            self.hashed[r] = self.hash(self.data[r])

    def exact_search(self, query, k=50):
        """
        Args:
            query: torch.Tensor query vector of shape (d,)
            k: how many top k results we want

        Returns:
            torch.Tensor of shape(k,) containing indices of [k] most
            similar rows of [self.hashed] in terms of cosine similarity
        """
        cosines = torch.nn.functional.cosine_similarity(query, self.data)
        cosines[list(self.removed_rows)] = -torch.inf
        return torch.topk(cosines, k).indices
    
    ########## UTILITY FUNCTIONS ##########

    def stats(self, exact_idxs, lsh_idxs):
        """
        Args:
            exact_idxs: torch.Tensor of shape (k,), the ground truth
            lsh_idxs: torch.Tensor of shape (k,), the predicted
        
        Returns:
            precision, recall, f1-score between [exact_idxs] and [lsh_idxs]
        """
        tp = len(set(exact_idxs.tolist()) & set(lsh_idxs.tolist()))
        precision = tp / len(lsh_idxs) if len(lsh_idxs) > 0 else 0
        recall = tp / len(exact_idxs) if len(exact_idxs) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0
        return precision, recall, f1
    
    def timer_start(self):
        """
        reset timer to current
        """
        self.started_time = datetime.now()

    def timer_measure(self) -> int:
        """
        Returns:
            elapsed time in ms
        """
        return (datetime.now() - self.started_time).total_seconds() * 1000

def experiment(file, num_bits, device):
    path = '../data/' + file + '.npy'

    data = torch.tensor(np.load(path), dtype=torch.float32).to(device)
    data = torch.nn.functional.normalize(data)

    lsh = LSH(num_bits, data.shape[1], device)
    
    lsh.timer_start()
    lsh.index(data)
    index_time = lsh.timer_measure()

    r = int(np.random.random() * data.shape[0])
    c = int(np.random.random() * data.shape[1])
    
    query = data[r]
    exact = lsh.exact_search(query)

    lsh.timer_start()
    approx = lsh.search(query, k=50)
    search_time = lsh.timer_measure()

    lsh.timer_start()
    lsh.unlearn(r, c, 0)
    unlearn_time = lsh.timer_measure()

    precision, _, _ = lsh.stats(exact, approx)

    return round(precision, 4), index_time, search_time, unlearn_time

def experiment_num_bits(dataset, num_trials=10):
    nbits = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    precisions = []
    index_times = []
    search_times = []
    unlearn_times = []

    for nbit in nbits:
        print('Number of Bits:', nbit)

        avg_stats = np.zeros(4)
        for iter in range(num_trials):
            print('Iter:', iter)
            avg_stats += experiment(dataset, nbit, 'mps')
        avg_stats /= num_trials

        precisions.append(avg_stats[0])
        index_times.append(avg_stats[1])
        search_times.append(avg_stats[2])
        unlearn_times.append(avg_stats[3])

    plt.plot(nbits, precisions)
    plt.xlabel('Number of LSH Hash Bits')
    plt.ylabel('Precision')
    plt.title('Precision vs Number of LSH Hash Bits on ' + dataset)
    plt.legend()
    plt.savefig('../results/final_precision_' + dataset + '.png')

    print(index_times)
    print(search_times)
    print(unlearn_times)

    # plt.clf()

    # plt.plot(nbits, index_times, label='Indexing Time')
    # plt.plot(nbits, search_times, label='Search TIme')
    # plt.xlabel('Number of LSH Hash Bits')
    # plt.ylabel('Runtime (ms)')
    # plt.title('Runtime vs Number of LSH Hash Bits on ' + dataset)
    # plt.legend()
    # plt.savefig('../results/final_precision_' + dataset + '.png')

if __name__ == "__main__":
    # precision, index_time, search_time, unlearn_time = experiment('lastfm', 1024, 'mps')

    # print(f"Precision: {precision:.4f}")
    # print('Index Time:', index_time)
    # print('Search Time:', search_time)
    # print('Unlearn Time:', search_time)

    experiment_num_bits('lastfm')