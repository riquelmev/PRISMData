from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import ReLU
from keras.layers import Activation
from keras.layers import Lambda
import keras.backend as K
from collections import Counter
import json
import tensorflow as tf
import tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers import activation
from sklearn import metrics
from sklearn.metrics import classification_report
from keras.metrics import MeanSquaredError
import keras
import pickle as pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
from keras import backend as K
import tensorflow as tf

#
# file = 'xtraining_data_SMOTEDown.pickle'
# with open(file, 'rb') as f:
#     x_train = pickle.load(f)
#
# file = 'ytraining_data_SMOTEDown.pickle'
# with open(file, 'rb') as f:
#     y_train = pickle.load(f)

# file = '/home/vicente/storage/xtraining_data_SMOTE0.pickle'
# with open(file, 'rb') as f:
#     x_train = pickle.load(f)
#
# file = '/home/vicente/storage/ytraining_data_SMOTE0.pickle'
# with open(file, 'rb') as f:
#     y_train = pickle.load(f)

# print(y_train)
# y_val_class = [np.argmax(y, axis=None, out=None) for y in y_train]
# new_y_val = []
# for val in y_val_class:
#     if val > 0:
#         new_y_val.append(1)
#     else:
#         new_y_val.append(0)
# print(new_y_val)
# y_val = np.asarray(new_y_val)
# y_val = tf.keras.utils.to_categorical(y_val)

file = '/mount/storage/data/x_validationbins.pickle'
with open(file, 'rb') as f:
    x_val = pickle.load(f)

file = '/mount/storage/data/y_validation_4catagories.pickle'
with open(file, 'rb') as f:
    y_val = pickle.load(f)

print(x_val.shape)
print(y_val.shape)
#print(y_train)
#print(y_val)
# y_val_class = [np.argmax(y, axis=None, out=None) for y in y_val]
#
#
# new_y_val = []
# for val in y_val_class:
#     if val > 0:
#         new_y_val.append(1)
#     else:
#         new_y_val.append(0)
# y_val = np.asarray(new_y_val)
#
# y_val = tf.keras.utils.to_categorical(y_val)
#print(y_val)
#print(y_val.shape)
#print("STOP")
#y_classes = [np.argmax(y, axis=None, out=None) for y in y_train]

# bins = [[]] * 4
#
# print(y_classes.count(0))
# print(y_classes.count(1))
# print(y_classes.count(2))
# print(y_classes.count(3))
#
# print(y_train)
# new_y_train = []
# for val in y_classes:
#     if val > 0:
#         new_y_train.append(1)
#     else:
#         new_y_train.append(0)
# y_train = np.asarray(new_y_train)
# y_train = tf.keras.utils.to_categorical(y_train)
# print(y_train)
# print(y_train.shape)

#print(y_classes)
# for num in :
#     bins[num].append(num)
# for i in range(4):
#     print(len(bins[i]))
batch_size = 512
input_shape = (20,8,1)
model = Sequential()
model.add(Reshape(input_shape))
#model.add(Dropout(0.2))
model.add(Conv2D(1024, (3,8), 1, activation='relu', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(512, (2,1), 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(Conv2D(256, (2,1), 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(128, (2,1),1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
#model.add(Conv2D(64, (2,1), activation='relu'))
model.add(Conv2D(64, (2,1),1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(Conv2D(32, (2,1),1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(Flatten())

#Change depending on predictions
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(4, activation='softmax'))

#top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=2)
#top3_acc.__name__ = 'top3_acc'
#model.compile(loss=keras.losses.MeanSquaredLogarithmicError(), optimizer=opt)
#model.compile(optimizer="Adam", loss=keras.losses.MeanSquaredLogarithmicError(), metrics=["mae", "acc"])
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=["mse", "acc"])

#print(x_train.shape)
#print(x_val.shape)
#print(y_val.shape)
#print(y_train.shape)

for i in range(60):
    print(i)
    xfile = '/mount/storage/data/xtraining_data_SMOTE' + str(i%20) + 'bins.pickle'
    yfile = '/mount/storage/data/ytraining_data_SMOTE' + str(i%20) + 'bins.pickle'
    with open(xfile, 'rb') as f:
        x_train = pickle.load(f)
    with open(yfile, 'rb') as f:
        y_train = pickle.load(f)
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=True)
    if i%20 == 0:
        model.save('/mount/storage/4classmodel' + str(i))
    #model.evaluate(x_val, y_val)

model.evaluate(x_val,y_val)
model.summary()

#pred = model.predict(x_val)
y_pred_ohe = model.predict(x_val)  # shape=(n_samples, 12)
# only necessary if output has one-hot-encoding, shape=(n_samples)
y_pred_labels = np.argmax(y_pred_ohe, axis=1)
y_cat = np.argmax(y_val, axis=1)
print(y_pred_labels)
print(y_cat)

matrix = metrics.confusion_matrix(y_cat, y_pred_labels)
#matrix = metrics.confusion_matrix(y_val.argmax(axis=1), pred.argmax(axis=1))

print(matrix)

print(classification_report(y_cat,y_pred_labels))


