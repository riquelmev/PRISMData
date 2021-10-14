import collections
import tensorflow as tf
import pickle as pickle
import numpy as np
from sklearn import preprocessing
xholder =[]
yholder =[]
offset = 10
for i in range(5):
    j = i + offset
    print("Starting loop",j)
    file = '/home/vicente/storage/data/x_no_loss' + str(j) +'50msListFilter.pickle'
    with open(file, 'rb') as f:
        x_no_loss = pickle.load(f)
        print(" x no loss",x_no_loss.shape)

    file = '/home/vicente/storage/data/y_no_loss_data' + str(j) +'50msListFilter.pickle'
    with open(file, 'rb') as f:
        y_no_loss = pickle.load(f)
        print("y no loss",y_no_loss.shape)


    file = '/home/vicente/storage/data/x_loss_data50msListFilter.pickle'
    with open(file, 'rb') as f:
        x_loss = pickle.load(f)
        print("x loss",x_loss.shape)
    #print("xloss",x_loss.shape)
    file = '/home/vicente/storage/data/y_loss_data50msListFilter.pickle'
    with open(file, 'rb') as f:
        y_loss = pickle.load(f)
    print(x_loss.shape)
    print(y_loss.shape)
    for k in range(2):
        x_no_loss = np.concatenate((x_no_loss, x_loss))
        y_no_loss = np.concatenate((y_no_loss, y_loss))
    y_train = tf.keras.utils.to_categorical(y_no_loss, 2)

    with open('/home/vicente/storage/data/ytraining_data_Dup' + str(j) + '_50msListFilter.pickle', 'wb') as f:
        pickle.dump(y_train, f)


    with open('/home/vicente/storage/data/xtraining_data_Dup' + str(j) + '_50msListFilter.pickle', 'wb') as f:
        pickle.dump(x_no_loss, f)

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
