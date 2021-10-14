import pickle
import tensorflow as tf
import numpy as np

file = '/home/vicente/storage/data/y_validation2cat50ms.pickle'
with open(file, 'rb') as f:
    y_train = pickle.load(f)

file = '/home/vicente/storage/data/x_validation50ms.pickle'
with open(file, 'rb') as f:
    x_train = pickle.load(f)

print(x_train.shape)
print(y_train.shape)
print(y_train)
new_y_train = []

# for val in y_train:
#     if val > 0:
#         new_y_train.append(1)
#     else:
#         new_y_train.append(0)

bins = [[]] * 4

print(y_train)
#y_train = np.asarray(new_y_train)
y_train = tf.keras.utils.to_categorical(y_train,2)

print(y_train)
print(y_train.shape)

with open("/home/vicente/storage/data/y_validation_2catagories50ms.pickle", 'wb') as f:
    pickle.dump(y_train, f)