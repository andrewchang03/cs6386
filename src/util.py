from datetime import datetime
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

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
    
def create_random(n, d, density, normalize=True, seed=42):
    """
    Creates (n, d) data with density=[density] (in range [0, 1])
    """
    data = sparse.random(n, d, density=density, format='csr', random_state=seed)
    data = data.toarray().astype('float32')
    if normalize:
        data = normalize(data)
    return data

def normalize(data):
    data = data.astype(float)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    data /= norms
    return data

def sklearn_flat(q, data, k) -> np.ndarray:
    """ 
    Return top-k indices using sklearn cosine similarity

    Args:
        q: (m, d) ndarray matrix, set of query vectors
        data: (n, d) ndarray matrix
        k: int, how many top k elements

    Returns:
        (m, k) ndarray with each row containing top k indices
    """
    cosines = cosine_similarity(q, data)
    top_k = np.argsort(cosines)
    top_k = top_k[:,::-1]
    top_k = top_k[:,:k]
    return top_k

def stats(exact: set, estimate: set):
    """
    Returns precision, recall, f-score on [exact] ground truth set and
    predicted [estimate]
    """
    tp = len(exact & estimate)
    precision = tp / len(estimate)
    recall = tp / len(exact)
    f = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f

def avg_over_five(func, *args, **kwargs):
    """
    Run 5 trials of [func] and return average precision, recall, and f-score
    """
    total = 0
    precision_total = recall_total = f_total = 0
    for _ in range(5):
        p, r, f = func(*args, **kwargs)
        precision_total += p
        recall_total += r
        f_total += f
    precision_avg = round(precision_total / 5, 5)
    recall_avg = round(recall_total / 5, 5)
    f_avg = round(f_total / 5, 5)
    return precision_avg, recall_avg, f_avg

def absolute_path(base, *paths):
    return os.path.join(base, *paths)

def load_npz_dense(path, normalize=True):
    """
    Loads npz file (csr matrix) from [path] 
    and returns densified np.ndarray of it
    """
    data = sparse.load_npz(path).toarray().astype('float32')
    if normalize:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        data /= norms
    return data