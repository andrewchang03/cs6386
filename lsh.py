import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

seed = 42
np.random.seed(seed)

def compute_cosines(q: sparse.csr_matrix, data: sparse.csr_matrix):
    """
    Computes the cosine similarities between [q] and each row of [data]
    """
    cosines = cosine_similarity(q, data)
    return cosines.flatten()

def query_exact_top_k(q: sparse.csr_matrix, data: sparse.csr_matrix, k: int):
    """
    Retrieves indices of top [k] similar rows of [data] to [q] based on cosine similarity.
    Returns numpy array of pairs containing (index, similarity)
    """
    cosines = compute_cosines(q, data)
    top_k = np.argsort(cosines)[:len(cosines) - k - 1:-1]
    top_k = np.array([(i, cosines[i]) for i in top_k])
    return top_k

def random_hyperplanes(n_bits: int, d: int):
    """
    Generate [n_bits] random hyperplanes in [d] dimension
    """
    hyperplanes = np.random.randn(n_bits, d)
    return hyperplanes

def query_hash(q: sparse.csc_matrix, hyperplanes: np.ndarray, power_of_twos: np.ndarray):
    dot_products = q @ hyperplanes.T
    dot_products = dot_products > 0
    hash = np.dot(dot_products, power_of_twos)
    return hash

def generate_hash_buckets(data: sparse.csr_matrix, hyperplanes: np.ndarray, power_of_twos: np.ndarray):
    dot_products = data @ hyperplanes.T
    dot_products = dot_products > 0
    hashes = np.dot(dot_products, power_of_twos)
    buckets = defaultdict(list)
    for i in range(len(hashes)):
        buckets[hashes[i]].append(i)
    return buckets

if __name__ == "__main__":
    data = sparse.random(50, 100, density=0.1, format='csr', random_state=seed)
    q = sparse.random(1, 100, density=0.1, format='csr', random_state=seed)
    k = 5
    n_bits = 4
    power_of_twos = 2 ** np.arange(n_bits - 1, -1, -1)

    exact_top_k = query_exact_top_k(q, data, k)
    print(exact_top_k)

    hyperplanes = random_hyperplanes(n_bits, data.shape[1])
    buckets = generate_hash_buckets(data, hyperplanes, power_of_twos)
    hash = query_hash(q, hyperplanes, power_of_twos).item()
    print(buckets[hash])