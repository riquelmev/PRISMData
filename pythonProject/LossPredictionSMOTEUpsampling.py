import collections

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import tensorflow as tf
import pickle as pickle
import numpy as np
from sklearn import preprocessing

for i in range(20):
    #i = i + 10
    print("Starting loop",i)
    file = '/home/vicente/storage/data/x_no_loss' + str(i) +'50ms.pickle'
    with open(file, 'rb') as f:
        x_no_loss = pickle.load(f)

    file = '/home/vicente/storage/data/y_no_loss_data' + str(i) +'50ms.pickle'
    with open(file, 'rb') as f:
        y_no_loss = pickle.load(f)

    file = '/home/vicente/storage/data/x_loss_data50ms.pickle'
    with open(file, 'rb') as f:
        x_loss = pickle.load(f)

    file = '/home/vicente/storage/data/y_loss_data50ms.pickle'
    with open(file, 'rb') as f:
        y_loss = pickle.load(f)

    #print(y_loss)
    #print(y_no_loss)
    print("shapes")
    print(x_no_loss.shape)
    print(x_loss.shape)

    print(y_no_loss.shape)
    print(y_loss.shape)

    x_train = np.concatenate((x_loss, x_no_loss))
    print(x_train.shape)

    x_loss = None
    x_no_loss = None


    y_train = np.concatenate((y_loss, y_no_loss))
    y_loss = None
    y_no_loss = None

    print(y_train.shape)
    #print(y_train)
    new_y_train = []
    print(collections.Counter(y_train))

    # for val in y_train:
    #     if val > 3:
    #         new_y_train.append(3)
    #     elif val > 1:
    #         new_y_train.append(2)
    #     elif val > 0:
    #         new_y_train.append(1)
    #     else:
    #         new_y_train.append(0)

    bins = [[]] * 4
    for val in new_y_train:
        if val == 3:
            print(val)
    #print(y_train)
    #y_train = np.asarray(new_y_train)
    #y_train = preprocessing.LabelEncoder().fit_transform(y_train)
    #print(y_train)
    #print(y_train)
    #print(y_train.shape)
    y_loss = None
    y_no_loss = None

    load = None
    #print(type(x_train))
    #print(x_train.shape)
    #print(y_train.shape)

    orig_shape = x_train.shape
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)
    print("done resampling")
    #print(x_train.shape)
    #print(y_train.shape)
    orig_shape = x_train.shape[0], 20, 9
    x_train = np.reshape(x_train, orig_shape)
    #print(x_train.shape)

    print(y_train)
    print(collections.Counter(y_train))
    ytrain = np.asarray(y_train)
    y_train = tf.keras.utils.to_categorical(y_train,2)
    #new_y_train = []

    # for val in y_train:
    #     if val == [1]:
    #         new_y_train.append(1)
    #     else:
    #         new_y_train.append(0)
    #y_classes = [np.argmax(y, axis=None, out=None) for y in y_train]

    #y_train = np.asarray(new_y_train)
    #y_train = tf.keras.utils.to_categorical(y_train,2)

    print(y_train)
    with open('/home/vicente/storage/data/ytraining_data_SMOTE' + str(i) + '50ms.pickle', 'wb') as f:
        pickle.dump(y_train, f)

    with open('/home/vicente/storage/data/xtraining_data_SMOTE' + str(i) +'50ms.pickle', 'wb') as f:
        pickle.dump(x_train, f)


    print(y_train)
    x_train = 0
    y_train = 0

