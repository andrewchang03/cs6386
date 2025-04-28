import numpy as np
from collections import defaultdict
from scipy import sparse
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import caboose
import pandas as pd
import util

class Timer():
    """
    Timer class. start() to reset timer, measure() to stop and get time elapsed
    """
    def __init__(self):
        self.current = datetime.now()

    def start(self):
        self.current = datetime.now()

    def measure(self) -> int:
        return (datetime.now() - self.current).total_seconds() * 1000

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

def csr_random_nonzero(data):
    row_indices, col_indices = data.nonzero()
    rand_idx = np.random.choice(data.nnz)

    row = row_indices[rand_idx]
    col = col_indices[rand_idx]
    return row, col

def caboose_experiment(file, num_trials=10):
    path = '../data/' + file + '.npz'
    data = sparse.load_npz(path)
    n, d = data.shape

    index_time = search_time = unlearn_time = 0

    for iter in range(num_trials):
        timer = Timer()
        index = caboose.Index(n, d, data.indptr, data.indices, data.data, 50)
        index_time += timer.measure()

        timer.start()
        index.topk(int(np.random.random() * n))
        search_time += timer.measure()

        r, c = csr_random_nonzero(data)
        timer.start()
        index.forget(r, c)
        unlearn_time += timer.measure()

    index_time /= num_trials
    search_time /= num_trials
    unlearn_time /= num_trials

    return index_time, search_time, unlearn_time

def experiment(data, num_bits, device):
    data = torch.tensor(data, dtype=torch.float32).to(device)
    data = torch.nn.functional.normalize(data)

    timer = Timer()
    lsh = LSH(num_bits, data.shape[1], device)
    
    timer.start()
    lsh.index(data)
    index_time = timer.measure()

    r = int(np.random.random() * data.shape[0])
    c = int(np.random.random() * data.shape[1])
    
    query = data[r]
    exact = lsh.exact_search(query)

    timer.start()
    approx = lsh.search(query, k=50)
    search_time = timer.measure()

    timer.start()
    lsh.unlearn(r, c, 0)
    unlearn_time = timer.measure()

    precision, _, _ = lsh.stats(exact, approx)

    return round(precision, 4), index_time, search_time, unlearn_time

def experiment_num_bits(dataset, num_trials=10):
    path = '../data/' + dataset + '.npy'
    data = np.load(path)

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
            avg_stats += experiment(data, nbit, 'mps')
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

def gen_runtime_graphs():
    # index, search, unlearn
    # 0.64, 0.77, 0.59, 0.57
    # 2048 bits
    lastfm_caboose = [89.7826, 0.0191, 0.6436]
    lastfm_cpu = [6.912131e+02, 3.480790e+01, 3.151930e+01]
    lastfm_mps = [1.1045, 2.5019, 0.2714]

    movie_caboose = [36741.9109, 0.0207, 19.8799]
    movie_cpu = [3.0132196e+03, 7.9627500e+01, 2.2448500e+00]
    movie_mps = [16.1939, 58.7286,  0.429 ]

    # Grouping data
    labels = ['First Element', 'Second Element', 'Third Element']

    # Create DataFrame
    data = {
        'Caboose': [round(val, 2) for val in lastfm_caboose],
        'CPU': [round(val, 2) for val in lastfm_cpu],
        'MPS': [round(val, 2) for val in lastfm_mps]
    }

    df = pd.DataFrame(data, index=labels)

    # Display table
    print(df)

if __name__ == "__main__":
    # gen_runtime_graphs()

    # path = '../data/movie.npy'
    # data = np.load(path)

    # avg_stats = np.zeros(4)

    # for iter in range(20):
    #     print(iter)
    #     avg_stats += experiment(data, 2048, 'mps')

    # avg_stats /= 20
    # print(avg_stats)

    # precision, index_time, search_time, unlearn_time = experiment(data, 2048, 'mps')
    
    # print(f"Precision: {precision:.4f}")
    # print('Index Time:', index_time)
    # print('Search Time:', search_time)
    # print('Unlearn Time:', search_time)

    experiment_num_bits('movie')

    # index_time, search_time, unlearn_time = caboose_experiment('movie')
    # print(index_time, search_time, unlearn_time)