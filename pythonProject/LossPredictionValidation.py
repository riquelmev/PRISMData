import pickle as pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
import pylab as pl
from tqdm import tqdm
import json




import tensorflow as tf

import pickle as pickle
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import argmax
import pylab as pl
from tqdm import tqdm
import json
#file = '/SNAPPcapAnalysis/NNdataMlabs.pickle'
# file = 'NNdataMlabs.pickle'
# with open(file, 'rb') as f:
#     load = pickle.load(f)
#
# print("loaded")
#file = '/home/vicente/PycharmProjects/pythonProject/RTTDATA/RTTOUT_1626295365.jsonl'





x_val = []
y_val= []

#print(load[0][0])
#print(load[0][1])
bins = []
for i in range(3):
    bins.append([])
count = 0
total_loss = 0
total = 0

x_loss = []
y_loss =[]

x_no_loss = []
y_no_loss = []
#IN_FILE = "NNdataMlabs.jsonl"
IN_FILE = '/home/vicente/storage/JsonlMasterTesting.jsonl'


# fpath = os.path.join(IN_DIR, IN_FILE)
import jsonlines
# for line in open(IN_FILE).readlines():
#     if line != '\n':
with jsonlines.open(IN_FILE) as f:
    for line in f.iter():
        if line != '\n':
            window = line
            trace = window[1]
            if count % 10000 == 0:
                print(count)
            count +=1

            window = window[0]
            if np.isnan(np.sum(np.array(window[0]))) or np.isnan(np.sum(np.array(window[1]))):
                continue
            # if count < 6000000:
            #     continue
            #x_raw.append(window[0])
            lossRate = 0.0
            if window[1] == 0:# and count > 6000000:
                #y_raw.append(0)
                x_val.append(window[0])
                y_val.append(0)
                total = total +1
            elif window[1] > 0:# and count > 6000000:
                x_val.append(window[0])
                lossInterval = round(window[1], 3)
                lossInterval = int(lossInterval * 100)
                y_val.append(1)
                # if lossInterval < 1:
                #     y_val.append(1)
                # elif lossInterval < 3:
                #     y_val.append(2)
                # else:
                #     y_val.append(3)
            #window = None
    #y_raw.append(window[1])

x_val = np.asarray(x_val)
y_val = np.asarray(y_val)

num_classes = 4
#y_loss = tf.keras.utils.to_categorical(y_loss, num_classes)

print("starting pickle")
with open("/home/vicente/storage/data/x_validation50ms.pickle", 'wb') as f:
    pickle.dump(x_val, f)
#size.append(x_no_loss.shape[0])
x_no_loss = 0

with open("/home/vicente/storage/data/y_validation2cat50ms.pickle", 'wb') as f:
    pickle.dump(y_val, f)
#size.append(x_no_loss.shape[0])
y_no_loss= 0