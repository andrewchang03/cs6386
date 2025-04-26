import numpy as np

def read_data(fname, d=784):
    path = "../data/" + fname
    data = []
    labels = []

    with open(path, 'r') as file:
        rows = [line.rstrip() for line in file]
        for features in rows:
            features = features.split(' ')
            label = features.pop(0)
            labels.append(label)

            indices = list(map(lambda feature: int(feature.split(':')[0]), features))
            features = list(map(lambda feature: int(feature.split(':')[1]), features))

            row = []

            for i in range(d):
                if len(indices) > 0 and i == indices[0]:
                    indices.pop(0)
                    val = features.pop(0)
                    row.append(val)
                else:
                    row.append(0)
            
            data.append(row)
    
    data = np.array(data)

    with open('../data/mnist.npy', 'wb') as f:
        np.save(f, data)

    return data

if __name__ == "__main__":
    # read_data("mnist")
    # mnist = np.load('../data/mnist.npy')
    pass