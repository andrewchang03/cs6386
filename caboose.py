import numpy as np
import similaripy as sim
import scipy.sparse as sps
import heapq
import bisect
import copy

def build_index(data, k):
    norms = np.sqrt(data.multiply(data).sum(axis=1))
    model = sim.cosine(data, k=k+1, verbose=False)  # generate top-k
    index = [[] for _ in range(model.shape[0])]

    for r, c, val in zip(model.row, model.col, model.data):
        if r == c:
            continue
        heapq.heappush(index[r], [val, c])
        if len(index[r]) > k:
            heapq.heappop(index[r])

    return norms, index

def forget(data, i, g, k, norms, index):
    norms[i] = np.sqrt(norms[i] ** 2 - data[i, g] ** 2)
    zeroed_candidates = data.getrow(i) @ data.T
    data[i, g] = 0

    dots = data.getrow(i) @ data.T
    cosines = [[(value / (norms[i] * norms[col])).item(), col] 
               for col, value in zip(dots.indices, dots.data) if col != i]
    for candidate in zeroed_candidates.indices:
        if candidate not in dots.indices:
            cosines.append([0, candidate])
    index[i] = copy.deepcopy(cosines)
    index[i] = heapq.nlargest(k, index[i])
    heapq.heapify(index[i])
    
    for cosine, r in cosines:
        heap = index[r]
        sorted_keys = [tup[1] for tup in heap]
        sorted_keys.sort()

        heapq._heapify_max(heap)
        top = heap[0][0]
        heapq.heapify(heap)
        
        pos = bisect.bisect_left(sorted_keys, i)
        found = (pos != len(sorted_keys) and sorted_keys[pos] == i)

        if not found:
            # print('case1', r)
            heapq.heappush(heap, [cosine, i])
            if len(heap) > k:
                heapq.heappop(heap)
        else:
            if cosine != 0:
                if cosine >= top:
                    # print('case2', r)
                    for tup in heap:
                        if tup[1] == i:
                            tup[0] = cosine
                    heapq.heapify(heap)
                else:
                    # print('case3', r)
                    rdots = data.getrow(r) @ data.T
                    rcosines = [[(value / (norms[r] * norms[col])).item(), col] 
                            for col, value in zip(rdots.indices, rdots.data) if col != r]
                    rcosines = heapq.nlargest(k, rcosines)
                    heapq.heapify(rcosines)
                    index[r] = rcosines
            else:
                if len(heap) < k:
                    # print('case4', r)
                    heap = list(filter(lambda tup: tup[1] != i, heap))
                    heapq.heapify(heap)
                    index[r] = heap
                else:
                    # print('case5', r)
                    rdots = data.getrow(r) @ data.T
                    rcosines = [[(value / (norms[r] * norms[col])).item(), col] 
                            for col, value in zip(rdots.indices, rdots.data) if col != r]
                    rcosines = heapq.nlargest(k, rcosines)
                    heapq.heapify(rcosines)
                    index[r] = rcosines
    
    return index
        
def ground_truth(data, i, g, k):
    data[i, g] = 0
    norms, index = build_index(data, k)
    return index

def index_equal(index1, index2, eps=0.01):
    counter = 0
    for heap1, heap2 in zip(index1, index2):
        heap1 = list(filter(lambda tup: tup[0] != 0, heap1))
        heap2 = list(filter(lambda tup: tup[0] != 0, heap2))
        if len(heap1) != len(heap2):
            print(counter, 'inequal length')
            print(heap1)
            print(heap2)
            return False
        heap1.sort(key=lambda tup: tup[0])
        heap2.sort(key=lambda tup: tup[0])
        for h1, h2 in zip(heap1, heap2):
            if abs(h1[0] - h2[0]) > eps or h1[1] != h2[1]:
                print(counter, 'value error')
                print(heap1)
                print(heap2)
                return False
        counter += 1
    return True

def compare(data, r, c, k):
    actual_index = ground_truth(copy.deepcopy(data), r, c, k)
    norms, index = build_index(data, k)
    unlearned_index = forget(copy.deepcopy(data), r, c, k, norms, copy.deepcopy(index))
    return index_equal(copy.deepcopy(actual_index), copy.deepcopy(unlearned_index))

data = sps.random(20, 40, density=0.1, format='csr', random_state=42)
k = 5

for r in range(data.shape[0]):
    for c in data.getrow(r).indices:
        if not compare(data, r, c, k):
            print(r, c)

# print(compare(data, 5, 18, k))