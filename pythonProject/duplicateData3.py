import collections

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import tensorflow as tf
import pickle as pickle
import numpy as np
from sklearn import preprocessing
xholder =[]
yholder =[]
offset = 0
name = offset//2
print(name)
for i in range(2):
    j = i + offset
    print("Starting loop",i)
    with open('/home/vicente/storage/data/ytraining_data_Dup' + str(j) + '_50msListFilter.pickle', 'rb') as f:
        y = pickle.load(f)

    with open('/home/vicente/storage/data/xtraining_data_Dup' + str(j) + '_50msListFilter.pickle', 'rb') as f:
        x = pickle.load(f)

    if i == 0:
        x_temp = x
        y_temp = y
    else:
        x_temp = np.concatenate((x_temp,x))
        y_temp = np.concatenate((y_temp,y))
with open('/home/vicente/storage/data/ytraining_data_Dup50msListFilterCondensed_part' + str(name) + '.pickle', 'wb') as f:
        pickle.dump(y_temp,f)

with open('/home/vicente/storage/data/xtraining_data_Dup50msListFilterCondensed_part' + str(name) + '.pickle', 'wb') as f:
        pickle.dump(x_temp,f)
