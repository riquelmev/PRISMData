import pickle as pickle

import keras.models
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
#import pylab as pl
from keras import backend as K
import tensorflow as tf
file = '/home/vicente/storage/data/x_validation50ms.pickle'
with open(file, 'rb') as f:
    x_val = pickle.load(f)

#file = '/mount/storage/data/y_validation_2catagories50ms.pickle'
file = '/home/vicente/storage/data/y_validation_2catagories50ms.pickle'
with open(file, 'rb') as f:
    y_val = pickle.load(f)

print(x_val.shape)
print(y_val.shape)
model = keras.models.load_model('/home/vicente/storage/2classmodelTest0.82')
model.evaluate(x_val, y_val)
y_pred_ohe = model.predict(x_val)  # shape=(n_samples, 12)
# only necessary if output has one-hot-encoding, shape=(n_samples)
#y_pred_labels = np.argmax(y_pred_ohe, axis=1)
y_cat = np.argmax(y_val, axis=1)
#print(y_pred_labels)
#print(y_cat)
print(y_pred_ohe)
with open("/home/vicente/storage/data/modelpredictions50msTest.pickle", 'wb') as f:
    pickle.dump(y_pred_ohe, f)
with open("/home/vicente/storage/data/modelactual50msTest.pickle", 'wb') as f:
    pickle.dump(y_cat, f)