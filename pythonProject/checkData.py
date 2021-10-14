import pickle
import numpy as np
import collections
for i in range(4):
    xfile = '/home/vicente/storage/data/x_condensed' + str(i%4) + '50ms.pickle'
    yfile = '/home/vicente/storage/data/y_condensed' + str(i%4) + '50ms.pickle'
    with open(xfile, 'rb') as f:
        x_train = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_train = pickle.load(f)

    y_classes = [np.argmax(y, axis=None, out=None) for y in y_train]
    print(collections.Counter(y_classes))

