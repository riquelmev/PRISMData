from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Lambda
import keras.backend as K
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import json
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from keras.metrics import MeanSquaredError
import keras
import pickle as pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
import pylab as pl
from tqdm import tqdm

file = 'xtraining_data.pickle'
with open(file, 'rb') as f:
    x_train = pickle.load(f)


file = 'ytraining_data.pickle'
with open(file, 'rb') as f:
    y_train = pickle.load(f)
# print(len(load))
# print(load)
# x_train = load[0]
# y_train = load[1]

load = None
print(type(x_train))
print(x_train.shape)
#x_split = np.array_split(x_train,5)
#y_split = np.array_split(y_train,5)

#x_train = x_split[0]

orig_shape = x_train.shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
print(x_train.shape)
print(orig_shape)
#counter = Counter(y_train)
#print(counter)
undersample = RandomUnderSampler(sampling_strategy='majority')
x_train, y_train = undersample.fit_resample(x_train, y_train)
print("done resampling")
orig_shape = x_train.shape[0], 20, 9
x_train = np.reshape(x_train, orig_shape)
print(x_train.shape)
print(y_train.shape)

#with open("xtraining_data.pickle", 'wb') as f:
#    pickle.dump(x_train, f)
with open("xtraining_data_SMOTEDown.pickle", 'wb') as f:
    pickle.dump(x_train, f)

with open("ytraining_data_SMOTEDown.pickle", 'wb') as f:
    pickle.dump(y_train, f)

#counter = Counter(y_train)
#print(counter)


#np.unique(y_train, return_counts=True)
#np.unique(y_val, return_counts=True)
#print(len(x_val))
#print(len(y_val))


#print(len(x_train))
#print(len(y_train))
#print(x_train[0])
#normalize?
#x_train = np.expand_dims(x_train, axis=-1)

