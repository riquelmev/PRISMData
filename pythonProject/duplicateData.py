import collections

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import tensorflow as tf
import pickle as pickle
import numpy as np
from sklearn import preprocessing
xholder =[]
yholder =[]
for i in range(5):
    j = i + 0
    print("Starting loop",i)
    file = '/home/vicente/storage/data/x_no_loss' + str(j) +'50msListFilter.pickle'
    with open(file, 'rb') as f:
        x_no_loss = pickle.load(f)
        print("no loss",x_no_loss.shape)

    file = '/home/vicente/storage/data/y_no_loss' + str(j) +'50msListFilter.pickle'
    with open(file, 'rb') as f:
        y_no_loss = pickle.load(f)

    if i == 0:
        x_train = x_no_loss
        y_train = y_no_loss
    if i > 0:
        x_train = np.concatenate((x_train,x_no_loss))
        y_train = np.concatenate((y_train,y_no_loss))
    print("final x no loss size",x_train.shape)

    print(y_train.shape)
file = '/home/vicente/storage/data/x_loss_data50ms.pickle'
with open(file, 'rb') as f:
    x_loss = pickle.load(f)
    print("x loss",x_loss.shape)
    #print("xloss",x_loss.shape)
file = '/home/vicente/storage/data/y_loss_data50ms.pickle'
with open(file, 'rb') as f:
    y_loss = pickle.load(f)
print(x_loss.shape)
print(y_loss.shape)
for i in range (20):
    x_train = np.concatenate((x_train, x_loss))
    y_train = np.concatenate((y_train, y_loss))
print(x_train.shape)
print(y_train.shape)
print(y_train)
y_train = tf.keras.utils.to_categorical(y_train, 2)

with open('/home/vicente/storage/data/ytraining_data_Dup0_50ms.pickle', 'wb') as f:
    pickle.dump(y_train, f)


with open('/home/vicente/storage/data/xtraining_data_Dup0_50ms.pickle', 'wb') as f:
    pickle.dump(x_train, f)

    #print(y_loss)
    #print(y_no_loss)
    print("shapes")
    # print(x_no_loss.shape)
    # print(x_loss.shape)
    #
    # print(y_no_loss.shape)
    # print(y_loss.shape)
    #
    # x_train = np.concatenate((x_loss, x_no_loss))
    # print(x_train.shape)
    # xholder.append(x_train)
    #
    # x_loss = None
    # x_no_loss = None
    #
    #
    # y_train = np.concatenate((y_loss, y_no_loss))
    # y_loss = None
    # y_no_loss = None