import faiss
import numpy as np
from scipy import sparse
from sklearn.random_projection import SparseRandomProjection
import matplotlib.pyplot as plt
import os

import util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_faiss_flat(data: np.ndarray):
    """
    Creates exact cosine similarity index on Faiss.
    Basically just storing the data.
    """
    index = faiss.IndexFlatIP(data.shape[1])
    index.add(data)
    return index

def create_faiss_lsh(data: np.ndarray, nbits: int):
    """
    Creates Faiss LSH index with nbits hash.
    """
    index = faiss.IndexLSH(data.shape[1], nbits)
    index.add(data)
    return index

def random_faiss_experiment(n, d, density, nbits, k):
    """
    Generate (n, d) random data of specified density.
    Compute top-k with Faiss flat and lsh index
    Return precision, recall, and f-score of LSH vs flat index (ground truth)
    """
    data = util.create_random(n, d, density, seed=None)

    flat = create_faiss_flat(data)
    lsh = create_faiss_lsh(data, nbits)

    q = util.create_random(1, d, density, seed=None)

    _, idx_flat = flat.search(q, k=k)
    _, idx_lsh = lsh.search(q, k=k)

    idx_flat = set(idx_flat.flatten())
    idx_lsh = set(idx_lsh.flatten())

    precision, recall, f = util.stats(idx_flat, idx_lsh)
    return precision, recall, f

def faiss_nbits(n, d, density, nbits, k):
    """
    Test Faiss over varying number of hash bits
    """
    nbits_x = [8, 16, 32, 64, 256, 512, 1024, 2048, 4096]
    nbits_y = []
    for x in nbits_x:
        p, _, _ = util.avg_over_five(
            random_faiss_experiment, n, d, density, x, k)
        nbits_y.append(p)
    
    # Plot
    plt.plot(nbits_x, nbits_y)
    plt.xlabel('Number of Hash Bits')
    plt.ylabel('Precision')
    plt.title('Faiss LSH Precision vs Number of Hash Bits')
    plt.savefig('../results/faiss_nbits.png')

def faiss_sparsity(n, d, density, nbits, k):
    """
    Test Faiss over varying sparsity (1 - density)
    """
    sparsity_x = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    sparsity_y = []
    for x in sparsity_x:
        p, _, _ = util.avg_over_five(
            random_faiss_experiment, n, d, 1 - x, nbits, k)
        sparsity_y.append(p)
    
    # Plot
    plt.plot(sparsity_x, sparsity_y)
    plt.xlabel('Data Sparsity')
    plt.ylabel('Precision')
    plt.title('Faiss LSH Precision vs Data Sparsity')
    plt.savefig('../results/faiss_sparsity.png')

def faiss_dimensions(n, d, density, nbits, k):
    """
    Test Faiss over varying data dimensions d
    """
    dim_x = [10, 100, 500, 1000, 5000, 10000]
    dim_y = []
    for x in dim_x:
        p, _, _ = util.avg_over_five(
            random_faiss_experiment, n, x, density, nbits, k)
        dim_y.append(p)
    
    # Plot
    plt.plot(dim_x, dim_y)
    plt.xlabel('Data Dimensionality (log)')
    plt.ylabel('Precision')
    plt.xscale('log')
    plt.title('Faiss LSH Precision vs Data Dimensionality')
    plt.savefig('../results/faiss_dimensionality.png')

def save_flat_topk(name, k=50):
    """
    Save exact top-[k] index of dataset [name]
    """
    data = util.load_npz_dense('../data/' + name + '.npz')
    flat = create_faiss_flat(data)
    _, idx_flat = flat.search(data, k=k)

    with open('../results/' + name + '_top50.npy', 'wb') as f:
        np.save(f, idx_flat)

def save_mnist_topk(k=50):
    """
    Save exact top-[k] index of mnist
    """
    data = np.load('../data/mnist.npy')
    data = util.normalize(data)
    flat = create_faiss_flat(data)
    _, idx_flat = flat.search(data, k=k)

    with open('../results/mnist_top50.npy', 'wb') as f:
        np.save(f, idx_flat)

def faiss_dataset(name, nbits, k=50, n_components=500, reduce=False, seed=42):
    """
    Evaluate Faiss LSH vs Flat on real datasets.
    [reduce] if you want to reduce dimensionality of data
    """
    data = util.load_npz_dense('../data/' + name + '.npz')
    idx_flat = np.load('../results/' + name + '_top50.npy')

    if reduce:
        srp = SparseRandomProjection(
            n_components=n_components, random_state=seed)
        data = srp.fit_transform(data)

    timer = util.Timer()
    lsh = create_faiss_lsh(data, nbits)
    index_elapsed = timer.measure()

    timer.start()
    _, idx_lsh = lsh.search(data, k=k)
    search_elapsed = timer.measure()

    precisions = []

    for i in range(len(idx_flat)):
        idx_flat_row = set(idx_flat[i].flatten())
        idx_lsh_row = set(idx_lsh[i].flatten())
        p, _, _ = util.stats(idx_flat_row, idx_lsh_row)
        precisions.append(p)

    avg_precision = round(sum(precisions) / len(precisions), 5)
    return avg_precision, index_elapsed, search_elapsed

def faiss_dataset_nbits(name, k=50):
    """
    Experiments on specified dataset [name] with varying lsh hash bits
    """
    nbits_x = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    nbits_y = []
    nbits_reduced_y = []
    nbits_index = []
    nbits_search = []

    for nbits in nbits_x:
        print(nbits)
        precision, index_time, search_time = faiss_dataset(name, nbits)
        precision_reduced, _, _ = faiss_dataset(name, nbits, reduce=True)
        nbits_y.append(precision)
        nbits_reduced_y.append(precision_reduced)
        nbits_index.append(index_time)
        nbits_search.append(search_time)

    plt.plot(nbits_x, nbits_y, label='Original')
    plt.plot(nbits_x, nbits_reduced_y, label='Projected')
    plt.xlabel('Number of LSH Hash Bits')
    plt.ylabel('Precision')
    plt.title('Precision vs Number of LSH Hash Bits on ' + name)
    plt.legend()
    plt.savefig('../results/faiss_' + name + '.png')

    print(nbits_index)
    print(nbits_search)

def faiss_mnist(nbits, k=50, n_components=500, reduce=False, seed=42):
    """
    Same as faiss_dataset but for mnist
    """
    data = np.load('../data/mnist.npy')
    idx_flat = np.load('../results/mnist_top50.npy')

    if reduce:
        srp = SparseRandomProjection(
            n_components=n_components, random_state=seed)
        data = srp.fit_transform(data)

    timer = util.Timer()
    lsh = create_faiss_lsh(data, nbits)
    index_elapsed = timer.measure()

    timer.start()
    _, idx_lsh = lsh.search(data, k=k)
    search_elapsed = timer.measure()

    precisions = []

    for i in range(len(idx_flat)):
        idx_flat_row = set(idx_flat[i].flatten())
        idx_lsh_row = set(idx_lsh[i].flatten())
        p, _, _ = util.stats(idx_flat_row, idx_lsh_row)
        precisions.append(p)

    avg_precision = round(sum(precisions) / len(precisions), 5)
    return avg_precision, index_elapsed, search_elapsed

if __name__ == "__main__":
    n = 1000
    d = 100
    density = 0.7
    nbits = 256
    k = 50

    # Faiss random experiments
    # faiss_nbits(n, d, density, nbits, k)
    # faiss_sparsity(n, d, density, nbits, k)
    # faiss_dimensions(n, d, density, nbits, k)
    
    # save_flat_topk('movie', k=50)
    # save_flat_topk('lastfm', k=50)
    # save_mnist_topk()

    # avg_precision_movie, ie, se = faiss_dataset('movie', 1024)
    # avg_precision_movie, ie, se = faiss_dataset('movie', 1024, reduce=True)
    
    # faiss_dataset_nbits('lastfm')
    # faiss_dataset_nbits('movie')