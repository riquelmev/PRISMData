from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import MaxPooling2D
from keras.layers import  AveragePooling2D
from keras.layers import Dropout
import gc
import keras
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers import ReLU
from keras.layers import Activation
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.applications
from keras.layers.normalization import BatchNormalization
#from tensorflow.keras.layers import activation
from sklearn import metrics
from sklearn.metrics import classification_report
from keras.metrics import MeanSquaredError

import pickle as pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
#import pylab as pl
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

file = '/mount/storage/data/x_validation50ms.pickle'
with open(file, 'rb') as f:
    x_val = pickle.load(f)

#file = '/mount/storage/data/y_validation_2catagories50ms.pickle'
file = '/mount/storage/data/y_validation_2catagories50ms.pickle'
with open(file, 'rb') as f:
    y_val = pickle.load(f)

print(x_val.shape)
print(y_val.shape)
#print(y_train)
#print(y_val)
#y_val = [np.argmax(y, axis=None, out=None) for y in y_val]
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
batch_size = 1024
input_shape = (20,9,1)
#model = VGG16(weights='imagenet', include_top=False, input_shape= (20,8,1))
#model = tf.keras.applications.DenseNet121(input_shape=(20,8,1), include_top=False, weights=None)
model = Sequential()
model.add(Reshape(input_shape))
# #model.add(Dropout(0.2))
model.add(Conv2D(64, (4,9), 1, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,1)))
#model.add(Dropout(0.3))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
# #model.add(MaxPooling2D(pool_size=(2, 1)))
# #model.add(Conv2D(1024, (2,1), 1, activation='relu'))
#model.add(Conv2D(256, (2,1), 1, activation='relu'))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Conv2D(128, (4,1), 1, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1)))
model.add(Dropout(0.1))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
# #model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Conv2D(256, (2,1),1, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,1)))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
# #model.add(Conv2D(64, (2,1), activation='relu'))
#was 5 before, now 2
model.add(Conv2D(256, (2,1),1, activation='relu'))
model.add(Dropout(0.1))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
#model.add(Conv2D(384, (2,1),1, activation='relu'))
#model.add(Conv2D(256, (2,1),1, activation='relu'))
#model.add(MaxPooling2D(pool_size=(1,1)))
# # model.add(BatchNormalization())
# # model.add(Activation("relu"))
model.add(Flatten())

#Change depending on predictions
# model.add(Dense(1536, activation= 'relu'))
# model.add(Dropout(0.4))
# model.add(Dense(1024, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(768, activation= 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dropout(0.3))
#model.add(Dense(384, activation= 'relu'))
#model.add(Dropout(0.3))
model.add(Dense(256, activation= 'relu'))
#model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#opt = keras.optimizers.Adam(learning_rate = 0.0001)
#top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=2)
#top3_acc.__name__ = 'top3_acc'
#model.compile(loss=keras.losses.MeanSquaredLogarithmicError(), optimizer=opt)
#model.compile(optimizer="Adam", loss=keras.losses.MeanSquaredLogarithmicError(), metrics=["mae", "acc"])
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=["mse", "acc"])
#print(x_train.shape)
#print(x_val.shape)
#print(y_val.shape)
#print(y_train.shape)
#model = keras.model.load_model('mount2classmodel60')
# for i in range(30):
#     print(i)
#     xfile = '/mount/storage/data/xtraining_data_SMOTE' + str(i%20) + '.pickle'
#     yfile = '/mount/storage/data/ytraining_data_SMOTE' + str(i%20) + '.pickle'
#     with open(xfile, 'rb') as f:
#         x_train = pickle.load(f)
#     with open(yfile, 'rb') as f:
#         y_train = pickle.load(f)
#     model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=True)
#     if i%20 == 0:
#         model.save('/mount/storage/2classmodel' + str(i))
#     #model.evaluate(x_val, y_val)
#model = keras.models.load_model('/mount/storage/2classmodelTest0.82')
for i in range(250):
    y_train = None
    x_train = None
    # with open('/mount/storage/data/ytraining_data_Dup50msListFilterCondensed_part'+ str(i%5) +'.pickle', 'rb') as f:
    #     y_train = pickle.load(f)
    # with open('/mount/storage/data/xtraining_data_Dup50msListFilterCondensed_part'+ str(i%5) +'.pickle', 'rb') as f:
    #     x_train = pickle.load(f)
    with open('/mount/storage/data/ytraining_data_Dup' + str(i%15) + '_50msListFilter.pickle', 'rb') as f:
        y_train = pickle.load(f)
    with open('/mount/storage/data/xtraining_data_Dup' + str(i%15) + '_50msListFilter.pickle', 'rb') as f:
        x_train = pickle.load(f)
    if i % 50 == 0 and i != 0:
        model.save('/mount/storage/2classmodelTestRound' + str(i))
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle= True)
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=1, batch_size=batch_size, shuffle=True)
    del y_train
    del x_train
    gc.collect()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_loss'])
# #plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('/mount/storage/graph.png')
# pd.DataFrame(history.history).plot(figsize=(8,5))
# plt.savefig('/mount/storage/graphall.png')
# for i in range(100):
#     print(i)
#     print(i%4)
#     print(str(i%4))
#     if i == 1:
#         model.summary()
#     #xfile = '/mount/storage/data/x_condensed' + str(i%4) + '50ms2.pickle'
#     #yfile = '/mount/storage/data/y_condensed' + str(i%4) + '50ms2.pickle'
#     xfile = '/mount/storage/data/xtraining_data_Dup' +str(i%5)+'_50msListFilter.pickle'
#     yfile = '/mount/storage/data/ytraining_data_Dup' +str(i%5)+'_50msListFilter.pickle'
#     with open(xfile, 'rb') as f:
#         x_train = pickle.load(f)
#     with open(yfile, 'rb') as f:
#         y_train = pickle.load(f)
#     #print(x_train.shape)
#     #print(x_train)
#     #print(y_train.shape)
#     #print(y_train)
#     #y_train = [np.argmax(y, axis=None, out=None) for y in y_train]
#     #xfile = np.expand_dims(xfile, axis=1)
#     #yfile = np.expand_dims(yfile, axis=1)
#     #print(xfile.shape)
#     #print(yfile.shape)
#     model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=True)
#if i%5 == 0:
 #+ str(i))
#     if i%5 == 0:
#         model.evaluate(x_val, y_val)
#         y_pred_ohe = model.predict(x_val)  # shape=(n_samples, 12)
#         # only necessary if output has one-hot-encoding, shape=(n_samples)
#         y_pred_labels = np.argmax(y_pred_ohe, axis=1)
#         y_cat = np.argmax(y_val, axis=1)
#         print(y_pred_labels)
#         print(y_cat)
#
#         matrix = metrics.confusion_matrix(y_cat, y_pred_labels)
#         # matrix = metrics.confusion_matrix(y_val.argmax(axis=1), pred.argmax(axis=1))
#
#         print(matrix)
#         # pl.matshow(matrix)
#         # pl.show()
#         print(classification_report(y_cat, y_pred_labels))

model.save('/mount/storage/2classmodelDUP')

model.evaluate(x_val,y_val)
#model.evaluate(x_valid,y_valid)
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
#pl.matshow(matrix)
#pl.show()
print(classification_report(y_cat,y_pred_labels))

# y_pred_ohe = model.predict(x_valid)  # shape=(n_samples, 12)
# # only necessary if output has one-hot-encoding, shape=(n_samples)
# y_pred_labels = np.argmax(y_pred_ohe, axis=1)
# y_cat = np.argmax(y_valid, axis=1)
# print(y_pred_labels)
# print(y_cat)
#
# matrix = metrics.confusion_matrix(y_cat, y_pred_labels)
# #matrix = metrics.confusion_matrix(y_val.argmax(axis=1), pred.argmax(axis=1))
#
# print(matrix)
# #pl.matshow(matrix)
# #pl.show()
# print(classification_report(y_cat,y_pred_labels))


